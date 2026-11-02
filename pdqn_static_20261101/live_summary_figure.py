"""Real sensor snapshots and observed geometry, never reconstructed positions."""
from pathlib import Path
import json
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    root=Path(__file__).parent/'early-live-case035'
    frames=[json.loads(s) for s in (root/'capture-timestamps.jsonl').read_text().splitlines()]
    rows=[json.loads(s) for s in (root/'decisions.jsonl').read_text().splitlines()]
    fig=plt.figure(figsize=(12,7.2),layout='constrained');gs=fig.add_gridspec(3,3,height_ratios=[1.35,1,1])
    for col,wanted in enumerate((0.,10.,29.9)):
        frame=min(frames,key=lambda f:abs(f['wall_s']-wanted))
        rgb=cv2.cvtColor(cv2.imread(str(root/f"frames/{frame['frame']:05d}.jpg")),cv2.COLOR_BGR2RGB)
        ax=fig.add_subplot(gs[0,col]);ax.imshow(rgb);ax.axis('off');ax.set_title(f"Actual RGB at {frame['wall_s']:.2f}s",fontsize=10)
    times=[r['sim_time'] for r in rows]
    ax=fig.add_subplot(gs[1,:2]);ax.plot(times,[r['center_error'] for r in rows],color='#2563eb',label='RGB center error')
    ax.axhline(.025,color='#15803d',ls='--',lw=1,label='hold mean threshold');ax.set(xlabel='Actual UE time (s)',ylabel='Normalized center error',xlim=(0,30));ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax=fig.add_subplot(gs[2,:2]);ax.plot(times,[r['area']*100 if r['area'] else None for r in rows],color='#c2410c',label='RGB box area')
    ax.axhline(4.,color='#15803d',ls='--',lw=1,label='4% target');ax.set(xlabel='Actual UE time (s)',ylabel='Box / image area (%)',xlim=(0,30));ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax=fig.add_subplot(gs[1:,2]);audit=json.loads((root/'independent-live-results.json').read_text());ax.axis('off')
    text=("Independent file audit\n\n"
          f"Last 5s center mean: {audit['center_mean']:.4f}\n"
          f"Last 5s center P95: {audit['center_p95']:.4f}\n"
          f"Last 5s area: {audit['area_mean']*100:.2f}%\n"
          "All final hold commands: Stay\n"
          f"Correct-target availability: {audit['matched_availability']:.0%}\n\n"
          "Known development example.\n"
          "Full development: 3 / 16 passed.\n"
          "Multi-case qualification is pending.")
    ax.text(.02,.96,text,va='top',fontsize=10,linespacing=1.6)
    fig.suptitle('Author-core P-DQN: actual real-time centering and active zoom',fontsize=14)
    fig.savefig(root/'actual-control-summary.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
