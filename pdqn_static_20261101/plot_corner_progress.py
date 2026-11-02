"""Compare every declared corner cell, including all failed candidates."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parent
    runs=[('320 episodes / 8,297 updates','audited-corner-development-0320'),
        ('400 episodes / support data','audited-corner-development-0400-support'),
        ('480 episodes / 11,238 updates','audited-corner-development-0480-support'),
        ('480 + 2,048 updates / old data','audited-corner-development-0480-replay')]
    for budget in (2048,1024,512):
        folder=f'audited-corner-development-exploration-u{budget:04d}'
        if (root/folder/'independent-results.json').exists():
            runs.append((f'480 + real fragments / +{budget:,} updates',folder))
    values=[];labels=[];evidence=[]
    for label,folder in runs:
        path=root/folder/'independent-results.json';report=json.loads(path.read_text())
        rows=report['episodes']
        if len(rows)!=32:continue
        cell=np.zeros(16,dtype=int)
        for row in rows:cell[row['scale_group']*4+row['quadrant']]+=int(row['passed'])
        values.append(cell);labels.append(f'{label}  ({sum(cell)}/32)')
        evidence.append(dict(report=str(path),checkpoint=report.get('configuration',{}).get('checkpoint'),passed=int(sum(cell)),cells=cell.tolist()))
    fig,ax=plt.subplots(figsize=(15,1.2+len(values)*.7),layout='constrained')
    cmap=ListedColormap(['#c75a55','#e8bd69','#4e9982']);norm=BoundaryNorm([-.5,.5,1.5,2.5],3)
    ax.imshow(values,cmap=cmap,norm=norm,aspect='auto')
    for (i,j),v in np.ndenumerate(np.array(values)):
        ax.text(j,i,f'{v}/2',ha='center',va='center',color='white' if v!=1 else '#3a3020',fontsize=10)
    ax.set(yticks=range(len(labels)),yticklabels=labels,xticks=range(16),
        xticklabels=['TL','TR','BL','BR']*4)
    ax.tick_params(axis='both',length=0)
    for x in (3.5,7.5,11.5):ax.axvline(x,color='white',lw=3)
    for j,name in enumerate(['Wide / smallest','Medium','Narrow','Largest initial']):
        ax.text(j*4+1.5,-.85,name,ha='center',fontsize=11)
    ax.set_title('Same 32 known development corner starts: two real rollouts per quadrant and scale',pad=40,fontsize=13)
    ax.set_xlabel('Green 2/2; amber 1/2; red 0/2. All failures retained. Development comparisons; heldout qualification is separate.',labelpad=12,fontsize=9)
    args.output.mkdir(exist_ok=False);fig.savefig(args.output/'corner-cell-comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)
    (args.output/'source-results.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print('Rendered',len(values),'complete candidate evaluations')

if __name__=='__main__':main()
