"""Boundary checks for independent acceptance, not simulated performance data."""
import json,copy
from pathlib import Path
from independent_validation import case

def main():
    out=Path(__file__).parent/'acceptance-boundaries';out.mkdir(exist_ok=True)
    initial=dict(initial=True,sim_time=0.,duration=0.,visible=True,action=0,confidence=.95,
        center_error=.01,x=.01,y=0.,area=.04,bbox=[.405,.4,.605,.6])
    rows=[initial]+[initial|dict(initial=False,sim_time=(i+1)*.1,duration=.1) for i in range(50)]
    variants={
        'valid_static_hold':(rows,True),
        'confidence_only_offset':([r|dict(x=.3,center_error=.3,bbox=[.55,.4,.75,.6]) for r in rows],False),
        'motion_during_hold':(rows[:-1]+[rows[-1]|dict(action=3)],False),
        'only_four_seconds':(rows[:41],False),
        'fabricated_geometry':(rows[:-1]+[rows[-1]|dict(center_error=0.)],False),
        'wrong_size':([r|dict(area=.01,bbox=[.455,.45,.555,.55]) for r in rows],False),
        'time_mismatch':(rows[:-1]+[rows[-1]|dict(duration=.2)],False)}
    checks={}
    for name,(samples,expected) in variants.items():
        folder=out/name;folder.mkdir(exist_ok=True)
        (folder/'initial.json').write_text(json.dumps(dict(seed=-1,index=-1,quadrant=0,layer=0,scale_group=0)))
        (folder/'steps.jsonl').write_text('\n'.join(json.dumps(r) for r in samples))
        (folder/'offline-diagnostics.jsonl').write_text('\n'.join(json.dumps(dict(sim_time=r['sim_time'],gt_bbox_normalized=r['bbox'])) for r in samples))
        result=case(folder);checks[name]=result['passed']==expected
        print(name,'PASS' if checks[name] else 'FAIL',result['failures'])
    report=dict(checks=checks,passed=all(checks.values()),scope='Synthetic boundary checks only; never count as UE episodes or training')
    (out/'checks.json').write_text(json.dumps(report,indent=2))
    assert report['passed']

if __name__=='__main__':main()
