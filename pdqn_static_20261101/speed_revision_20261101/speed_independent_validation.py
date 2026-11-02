"""Independent offline acceptance. It imports no environment or learner code."""
import argparse,csv,json,math
from pathlib import Path

def iou(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    intersection=x*y
    return intersection/max(1e-12,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)

def weighted_quantile(pairs,q):
    total=sum(w for v,w in pairs);cursor=0.
    for v,w in sorted(pairs):
        cursor+=w
        if cursor>=q*total:return v
    return None

def case(folder):
    rows=[json.loads(x) for x in (folder/'steps.jsonl').read_text().splitlines()]
    meta=json.loads((folder/'initial.json').read_text());end=rows[-1]['sim_time']
    errors=[];scales=[];areas=[];holds=[];loss=0.;loss_run=0.;max_loss=0.;previous={};reversals=0;overshoots=0;rapid=0
    valid_clock=True;valid_geometry=True;elapsed=0.;streak=0;source=False
    for before,row in zip(rows,rows[1:]):
        dt=row['sim_time']-before['sim_time'];elapsed+=dt
        minimum_duration=.002 if row['action'] in (1,2,3,4) else .06
        valid_clock &= dt>0 and abs(dt-row['duration'])<1e-5 and minimum_duration-1e-6<=row['duration']<=1.+1e-6
        if row['visible']:
            b=row.get('bbox')
            valid_geometry &= bool(b and len(b)==4 and b[2]>b[0] and b[3]>b[1])
            if b:
                bx,by=b[0]+b[2]-1,b[1]+b[3]-1
                valid_geometry &= abs(row['center_error']-math.hypot(bx,by))<1e-5
                valid_geometry &= abs(row['area']-(b[2]-b[0])*(b[3]-b[1]))<1e-5
        streak=streak+1 if row['confidence']>.9 else 0;source |= streak>=10
        if not row['visible']:loss+=dt;loss_run+=dt;max_loss=max(max_loss,loss_run)
        else:loss_run=0.
        overlap=max(0.,min(end,row['sim_time'])-max(end-5.,before['sim_time']))
        if overlap:
            holds.append((row,overlap))
            if row['visible']:
                errors.append((row['center_error'],overlap));areas.append((row['area'],overlap))
                scales.append((abs(math.log(row['area']/.04)),overlap))
        act=row['action']
        if act:
            axis=0 if act in (1,2) else 1 if act in (3,4) else 2
            if axis in previous and previous[axis][0]!=act:
                reversals+=1
                field='y' if axis==0 else 'x' if axis==1 else None
                if field and before['visible'] and row['visible']:
                    v0,v1=before[field],row[field]
                    if before['sim_time']-previous[axis][1]<=.5 and abs(v0)<=.03 and abs(v1)-abs(v0)>=.02:rapid+=1
            previous[axis]=(act,before['sim_time'])
            if before['visible'] and row['visible']:
                field='y' if axis==0 else 'x' if axis==1 else None
                if field and before[field]*row[field]<0 and min(abs(before[field]),abs(row[field]))>=.03:overshoots+=1
    avg=lambda pairs:sum(v*w for v,w in pairs)/sum(w for v,w in pairs) if pairs else None
    mean,p95,scale=avg(errors),weighted_quantile(errors,.95),avg(scales)
    hold_time=sum(w for r,w in holds);availability=1-loss/max(elapsed,1e-12)
    initial_box=rows[0].get('bbox')
    checks=dict(clock=valid_clock,geometry=valid_geometry,
        initial_full_detection=bool(rows[0].get('visible') and initial_box and all(.003<v<.997 for v in initial_box)),
        hold_duration=hold_time>=5.-1e-5,
        visible_during_hold=all(r['visible'] for r,w in holds),
        no_motion_during_hold=all(r['action']==0 for r,w in holds),
        center_mean=mean is not None and mean<=.025,center_p95=p95 is not None and p95<=.05,
        scale_mean=scale is not None and scale<=.25,
        area_band=bool(areas) and all(.025<=v<=.055 for v,w in areas),
        settling=end-5<=45.,target_retention=availability>=.99,
        maximum_unintended_loss=max_loss<=.25+1e-6,
        rapid_unwarranted_reversals=rapid<=2,
        reversal_frequency=reversals/max(elapsed,1e-12)*10<=4.+1e-6)
    # Simulator labels are consumed here, after control, and never supplied to policy.
    audit=folder/'offline-diagnostics.jsonl';matched_time=0.
    checks['decision_retention']=sum(r['visible'] for r in rows[1:])/max(1,len(rows)-1)>=.99
    checks['offline_gt_evidence']=audit.exists()
    if audit.exists():
        diagnostics=[json.loads(x) for x in audit.read_text().splitlines()]
        checks['paired_gt_count']=len(diagnostics)==len(rows)
        initial_gt=diagnostics[0].get('gt_bbox_normalized') if diagnostics else None
        checks['initial_gt_full_box']=bool(initial_gt and all(0.<=v<=1. for v in initial_gt))
        for before,row,diag in zip(rows,rows[1:],diagnostics[1:]):
            box=row.get('bbox');gt=diag.get('gt_bbox_normalized')
            checks['paired_gt_time']=checks.get('paired_gt_time',True) and abs(diag['sim_time']-row['sim_time'])<1e-5
            matched_time+=(row['sim_time']-before['sim_time'])*bool(box and gt and iou(box,gt)>=.3)
        checks['offline_matched_retention']=matched_time/max(elapsed,1e-12)>=.98
    return dict(seed=meta['seed'],index=meta['index'],quadrant=meta['quadrant'],layer=meta['layer'],scale_group=meta['scale_group'],
        passed=all(checks.values()),checks=checks,failures=[k for k,v in checks.items() if not v],
        source_confidence_condition=source,initial_center=rows[0]['center_error'],initial_area=rows[0]['area'],
        terminal_center_mean=mean,terminal_center_p95=p95,terminal_logscale_mean=scale,
        terminal_area_mean=avg(areas),settling_time=end-5,simulation_seconds=end,actions=len(rows)-1,
        availability=availability,max_loss_s=max_loss,reversals=reversals,overshoots=overshoots,rapid_unwarranted_reversals=rapid)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);args=p.parse_args()
    rows=[case(f.parent) for f in sorted(args.folder.glob('case-*/steps.jsonl'))]
    groups={}
    for name in ('quadrant','layer','scale_group'):
        groups[name]={}
        for value in sorted(set(r[name] for r in rows)):
            members=[r for r in rows if r[name]==value]
            groups[name][value]=dict(passed=sum(r['passed'] for r in members),total=len(members))
    report=dict(episodes=rows,groups=groups,passed=sum(r['passed'] for r in rows),count=len(rows),
        note='Independent offline metrics; unavailable GT evidence prevents qualification; source confidence and quality are separate')
    (args.folder/'independent-results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'},indent=2))
    if rows:
        with (args.folder/'independent-table.csv').open('w',newline='',encoding='utf-8-sig') as out:
            writer=csv.DictWriter(out,fieldnames=[k for k in rows[0] if k not in ('checks','failures')]);writer.writeheader()
            writer.writerows({k:v for k,v in row.items() if k not in ('checks','failures')} for row in rows)

if __name__=='__main__':main()
