"""Read-only analysis of real interactions; never modifies training or policy."""
from pathlib import Path
import argparse,json,math
import numpy as np

def read_lines(path):
    rows=[]
    if not path.exists():return rows
    for line in path.read_text(encoding='utf-8').splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass
    return rows

def analyze(root):
    episodes=read_lines(root/'episodes.jsonl');steps=[];updates=[]
    for path in sorted((root/'training').glob('*/steps.jsonl')):
        steps.extend(r for r in read_lines(path) if not r.get('initial'))
        updates.extend(read_lines(path.with_name('updates.jsonl')))
    seen=[r for r in steps if r['visible']]
    inner=[r for r in seen if r['center_error']<=.02 and abs(math.log(r['area']/.04))<=.18]
    bands=[r for r in seen if r['center_error']<=.05 and .025<=r['area']<=.055]
    mean=lambda key,rows:float(np.mean([r[key] for r in rows])) if rows else None
    result=dict(episodes=len(episodes),physical_steps=len(steps),valid_detection_steps=len(seen),
        inner_reward_steps=len(inner),broad_quality_steps=len(bands),
        success=sum(r['success'] for r in episodes),updates=len(updates),
        milliseconds={k:1000*mean(k,steps) for k in ('communication_s','capture_s','detector_s','reid_s','selection_s')} if steps else {},
        update_mean_ms=1000*mean('wall_s',updates[1:]) if len(updates)>1 else None,
        target_sync_updates=[u['update_count'] for u in updates if u['target_synchronised']],
        action_counts=np.bincount([r['action'] for r in steps],minlength=7).tolist(),
        executed_duration_quantiles=np.quantile([r['duration'] for r in steps],[0,.25,.5,.75,1]).tolist() if steps else [],
        raw_parameter_quantiles=np.quantile([r['raw_parameter'] for r in steps],[0,.01,.5,.99,1]).tolist() if steps else [],
        clipped_fraction=mean('execution_clipped',steps),
        initial_visible=sum(bool(json.loads(p.read_text())['visible']) for p in (root/'training').glob('*/initial.json')))
    print(json.dumps(result,indent=2))
    out=root/'analysis';out.mkdir(exist_ok=True)
    (out/'current.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if not episodes:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    x=np.arange(1,len(episodes)+1)
    for ax,key,label in ((axes[0,0],'reward','Episode return (exploration enabled)'),
                         (axes[0,1],'final_center','Final center error'),
                         (axes[1,0],'final_area','Final detected area / image area'),
                         (axes[1,1],'updates_total','Cumulative actual optimizer updates')):
        y=[r.get(key) if r.get(key) is not None else np.nan for r in episodes]
        ax.plot(x,y,alpha=.5,linewidth=.8);ax.set(xlabel='Training episode',ylabel=label);ax.grid(alpha=.2)
    axes[0,1].axhline(.025,color='green',linestyle='--',label='hold mean gate');axes[0,1].legend()
    axes[1,0].axhline(.04,color='green',linestyle='--',label='target');axes[1,0].legend()
    fig.suptitle('Training diagnostics only: no independent success claim')
    fig.tight_layout();fig.savefig(out/'learning.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();analyze(args.root)
