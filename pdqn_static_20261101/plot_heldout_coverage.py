"""Plot actual RGB initial detections for the completed heldout campaigns."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folders',type=Path,nargs=2,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');sources=[]
    for ax,folder in zip(axes,args.folders):
        report=json.loads((folder/'independent-results.json').read_text());assert len(report['episodes'])==160
        rows=[json.loads((folder/f'case-{i:03d}'/'initial.json').read_text()) for i in range(160)]
        seen=[r for r in rows if r['visible']]
        scatter=ax.scatter([(r['x']+1)*480 for r in seen],[(r['y']+1)*270 for r in seen],
            c=[r['area']*100 for r in seen],s=24,cmap='viridis',norm=LogNorm(.35,8.),alpha=.8,edgecolors='white',linewidths=.25)
        bad=[r for r in report['episodes'] if not r['passed']]
        for r in bad:
            initial=rows[r['index']]
            if initial['visible']:ax.scatter([(initial['x']+1)*480],[(initial['y']+1)*270],marker='x',s=70,color='#b91c1c')
        ax.axhline(270,color='gray',alpha=.25,lw=.7);ax.axvline(480,color='gray',alpha=.25,lw=.7)
        ax.set(xlim=(0,960),ylim=(540,0),xlabel='Actual detected start center x (pixels)',ylabel='Actual detected start center y (pixels)',
            title=f"{report['configuration']['split']}: {report['qualification']['passed']}/160 passed")
        sources.append(dict(report=str(folder/'independent-results.json'),actual_initials=160,visible=len(seen)))
    colorbar=fig.colorbar(scatter,ax=axes,label='Actual initial box / image area (%)',shrink=.8)
    colorbar.set_ticks([.5,1.,2.,4.,8.]);colorbar.set_ticklabels(['0.5','1','2','4','8'])
    fig.suptitle('One dot per real heldout case: 4 quadrants x 5 offset layers x 4 scales x 2',fontsize=13)
    fig.savefig(args.output,dpi=170);plt.close(fig)
    args.output.with_suffix('.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')

if __name__=='__main__':main()
