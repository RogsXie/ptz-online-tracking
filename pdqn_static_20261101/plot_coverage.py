"""Plot actual starting detections from unique UE case seeds."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',nargs='+',type=Path)
    p.add_argument('--output',required=True,type=Path);args=p.parse_args();rows={}
    for root in args.roots:
        for path in sorted(root.glob('training/*/initial.json')):
            row=json.loads(path.read_text());row['image_path']=str(path.with_name('initial.jpg'))
            rows[row['seed']]=row
    values=list(rows.values());seen=[r for r in values if r['visible']]
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
    ax=axes[0]
    scatter=ax.scatter([(r['x']+1)*480 for r in seen],[(r['y']+1)*270 for r in seen],
        c=[r['area']*100 for r in seen],s=30,cmap='viridis',norm=LogNorm(.35,8.),alpha=.85,edgecolors='white',linewidths=.3)
    ax.set(xlim=(0,960),ylim=(540,0),xlabel='Detected start center x (pixels)',ylabel='Detected start center y (pixels)',
        title=f'{len(values)} unique real UE start captures; {len(seen)} detected')
    ax.axvline(480,color='gray',alpha=.3);ax.axhline(270,color='gray',alpha=.3)
    fig.colorbar(scatter,ax=ax,label='Actual initial box area (%)',shrink=.8)
    counts=np.zeros((5,4),int)
    for r in values:counts[r['layer'],r['scale_group']]+=1
    ax=axes[1];im=ax.imshow(counts,cmap='Blues',vmin=0)
    for (i,j),v in np.ndenumerate(counts):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>counts.max()*.5 else '#102a43')
    ax.set(xticks=range(4),yticks=range(5),xticklabels=['wide','medium','narrow','large initial'],
        yticklabels=['center vicinity','near','inner','middle / edge','outer'],xlabel='Initial FOV / size group',ylabel='Offset layer',title='Actual captures per offset and size group')
    fig.suptitle('Initial coverage from RGB observations; no frames counted as separate scenes',fontsize=13)
    args.output.mkdir(exist_ok=True);fig.savefig(args.output/'actual-start-coverage.png',dpi=160);plt.close(fig)
    (args.output/'actual-starts.json').write_text(json.dumps(dict(unique_cases=len(values),visible=len(seen),rows=values),indent=2),encoding='utf-8')
    print('Unique actual starts',len(values),'visible',len(seen),'quadrants',[sum(r['quadrant']==q for r in values) for q in range(4)])

if __name__=='__main__':main()
