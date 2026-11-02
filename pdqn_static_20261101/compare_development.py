"""Identical known development cases, separately grouped; not final validation."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).parent

def main():
    paths=[ROOT/'training-source-cadence/development-0080/results.json',
           ROOT/'replay-budget-development/development-0080/results.json',
           ROOT/'online-interval4/development-0160/results.json']
    labels=['80 episodes\n289 updates','Same replay\n2337 updates','160 episodes\n4340 updates']
    runs=[json.loads(p.read_text()) for p in paths]
    latest=160
    for number in (240,320):
        candidate=ROOT/f'online-interval4/development-{number:04d}/results.json'
        if candidate.exists():
            rows=json.loads(candidate.read_text())
            if len(rows)==16:
                paths.append(candidate);runs.append(rows);labels.append(f'{number} episodes\n{rows[0]["updates_total"]} updates');latest=number
    assert all(sorted(r['seed'] for r in run)==sorted(r['seed'] for r in runs[0]) for run in runs)
    report=[]
    for label,run in zip(labels,runs):
        groups={}
        for name in ('quadrant','layer','scale_group'):
            groups[name]={}
            for v in sorted(set(r[name] for r in run)):
                members=[r for r in run if r[name]==v];groups[name][v]=dict(passed=sum(r['success'] for r in members),count=len(members))
        report.append(dict(label=label,passed=sum(r['success'] for r in run),count=len(run),groups=groups))
    out=ROOT/'development-comparison';out.mkdir(exist_ok=True)
    (out/f'through-{latest:04d}-results.json').write_text(json.dumps(dict(comparisons=report,scope='Same16known development cases; training stopping rule, no pairedGT side channel, no validation qualification'),indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
    values=[r['passed'] for r in report];bars=axes[0].bar(labels,values,color=['#94a3b8','#60a5fa','#2563eb'])
    for bar,value in zip(bars,values):axes[0].text(bar.get_x()+bar.get_width()/2,value+.35,f'{value}/16',ha='center')
    axes[0].set(ylim=(0,16.8),ylabel='Cases passing center / size / 5s Stay',title='Actual frozen development results');axes[0].grid(axis='y',alpha=.2)
    values=[report[-1]['groups']['quadrant'][q]['passed'] for q in range(4)]
    bars=axes[1].bar(['top left','top right','bottom left','bottom right'],values,color='#2563eb')
    for bar,value in zip(bars,values):axes[1].text(bar.get_x()+bar.get_width()/2,value+.1,f'{value}/4',ha='center')
    axes[1].set(ylim=(0,4.5),ylabel='Passed per quadrant',title=f'Remaining quadrant imbalance at episode{latest}');axes[1].tick_params(axis='x',labelrotation=20);axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Improvement is real; high-success independent acceptance is still pending',fontsize=12)
    fig.savefig(out/f'through-{latest:04d}-comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(report[-1],indent=2))

if __name__=='__main__':main()
