"""Synthetic feature-grid visualization only; never controls or trains UE."""
import bootstrap
from pathlib import Path
import argparse,json
import numpy as np,torch
from author_aligned.source import create_agent
from author_aligned.run import NETWORKS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2);agent=create_agent(20261101,'cpu')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    for n in NETWORKS:getattr(agent,n).load_state_dict(saved[n])
    x=np.linspace(-.12,.12,121);xx,yy=np.meshgrid(x,x)
    states=np.zeros((xx.size,8),np.float32);states[:,0]=xx.ravel();states[:,1]=yy.ravel();states[:,2]=4.;states[:,3]=.93
    with torch.inference_mode():
        parameters=agent.param_net(torch.from_numpy(states));q=agent.actor_net(torch.from_numpy(states),parameters)
        action=q.argmax(1);duration=parameters.gather(1,action[:,None]).clamp(.3,5.)/5.
    amap=action.numpy().reshape(xx.shape);dmap=duration.numpy().reshape(xx.shape)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    colors=['#cbd5e1','#3b82f6','#f97316','#10b981','#8b5cf6','#eab308','#ef4444']
    cmap=ListedColormap(colors);norm=BoundaryNorm(np.arange(-.5,7.5),7)
    im=axes[0].imshow(amap,extent=[-.12,.12,.12,-.12],cmap=cmap,norm=norm)
    bar=fig.colorbar(im,ax=axes[0],ticks=range(7));bar.ax.set_yticklabels(['Stay','Down','Up','Right','Left','Zoom in','Zoom out'])
    im2=axes[1].imshow(dmap,extent=[-.12,.12,.12,-.12],vmin=.06,vmax=1.,cmap='viridis')
    fig.colorbar(im2,ax=axes[1],label='Selected duration (s)')
    theta=np.linspace(0,2*np.pi,300)
    for ax in axes:
        ax.plot(.025*np.cos(theta),.025*np.sin(theta),'k--',lw=1)
        ax.set(xlabel='Image x offset (normalized)',ylabel='Image y offset (normalized)')
    axes[0].set_title('Original greedy action\nFixed area4%, confidence.93',fontsize=10)
    axes[1].set_title('Original parameter-network\nselected duration',fontsize=10)
    fig.suptitle(f"Offline synthetic-state diagnostic at episode{saved['completed_episodes']}; not UE validation",fontsize=11)
    args.output.parent.mkdir(exist_ok=True);fig.savefig(args.output,dpi=160);plt.close(fig)
    snapshot_path=args.output.parent/f"episode-{saved['completed_episodes']:04d}-networks.pt"
    if not snapshot_path.exists():
        torch.save({n:saved[n] for n in NETWORKS},snapshot_path)
    mask=amap==0;radius=np.hypot(xx,yy)
    report=dict(scope='Offline synthetic RGB-feature grid, no actual scenes or policy changes',checkpoint=str(args.checkpoint),
        completed=saved['completed_episodes'],network_snapshot=str(snapshot_path),stay_fraction_outside_acceptance=float(np.mean(mask[radius>.025])),
        maximum_stay_error=float(radius[mask].max()) if mask.any() else None,
        action_counts=np.bincount(amap.ravel(),minlength=7).tolist())
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
