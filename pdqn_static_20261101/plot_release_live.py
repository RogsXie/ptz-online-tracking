"""Real RGB snapshots and measured trajectories for one captured example."""
import argparse,json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);args=p.parse_args();root=args.folder
    frames=[json.loads(s) for s in (root/'capture-timestamps.jsonl').read_text().splitlines()]
    rows=[json.loads(s) for s in (root/'decisions.jsonl').read_text().splitlines()]
    audit=json.loads((root/'independent-release-live-results.json').read_text());cfg=json.loads((root/'configuration.json').read_text())
    fig=plt.figure(figsize=(12.5,7.6),layout='constrained');grid=fig.add_gridspec(3,3,height_ratios=[1.4,1.,1.])
    end=frames[-1]['wall_s']
    for column,wanted in enumerate((0.,min(10.,end/2),end)):
        frame=min(frames,key=lambda f:abs(f['wall_s']-wanted))
        rgb=cv2.cvtColor(cv2.imread(str(root/f"frames/{frame['frame']:05d}.jpg")),cv2.COLOR_BGR2RGB)
        ax=fig.add_subplot(grid[0,column]);ax.imshow(rgb);ax.axis('off');ax.set_title(f"Actual RGB at {frame['wall_s']:.2f} s",fontsize=10)
    times=[r['sim_time'] for r in rows];end_sim=times[-1]
    ax=fig.add_subplot(grid[1,:2]);ax.plot(times,[r['center_error'] for r in rows],color='#2563eb',lw=1.5)
    ax.axhline(.025,color='#15803d',ls='--',lw=1,label='5 s mean bound');ax.axvspan(end_sim-5,end_sim,color='#4e9982',alpha=.15)
    ax.set(xlabel='Actual UE time (s)',ylabel='RGB center error',xlim=(0,end_sim));ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax=fig.add_subplot(grid[2,:2]);ax.plot(times,[r['area']*100 if r['area'] else None for r in rows],color='#c2410c',lw=1.5)
    ax.axhline(4.,color='#15803d',ls='--',lw=1,label='4% target');ax.axvspan(end_sim-5,end_sim,color='#4e9982',alpha=.15)
    ax.set(xlabel='Actual UE time (s)',ylabel='Box / image area (%)',xlim=(0,end_sim));ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax=fig.add_subplot(grid[1:,2]);ax.axis('off')
    case=cfg['case'];settle=audit['settling_time_s'];settle_text=f'{settle:.2f} s' if settle is not None else 'not established'
    text=(f"Independent realtime audit: {'PASS' if audit['passed'] else 'FAIL'}\n\n"
        f"Final 5 s mean: {audit['center_mean']:.4f}\n"
        f"Final 5 s P95: {audit['center_p95']:.4f}\n"
        f"Final 5 s area: {audit['area_mean']*100:.2f}%\n"
        f"Settling: {settle_text}\n"
        f"All final hold commands Stay: {audit['checks']['all_stay']}\n"
        f"Correct-target availability: {audit['matched_availability']:.1%}\n\n"
        f"Raw RGB captures: {audit['raw_captured_frames']}\n"
        f"Encoded VFR frames: {audit['encoded_frames']}\n"
        f"Wall duration: {audit['wall_duration_s']:.2f} s\n\n"
        f"Known {case['split']} example {case['index']}.\n"
        "160-case final test is reported separately.")
    ax.text(.02,.97,text,va='top',fontsize=10,linespacing=1.5)
    fig.suptitle('Frozen author-core P-DQN: newly captured realtime control',fontsize=14)
    fig.savefig(root/'actual-control-summary.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
