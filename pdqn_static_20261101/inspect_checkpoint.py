"""Read-only CPU inspection of real replay; no rollout or training claim."""
import bootstrap
from bootstrap import HERE
from pathlib import Path
import argparse,json
import numpy as np,torch
from author_aligned.source import create_agent
from author_aligned.run import NETWORKS

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);args=p.parse_args()
    torch.set_num_threads(2)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    agent=create_agent(20261101,'cpu')
    for n in NETWORKS:getattr(agent,n).load_state_dict(saved[n])
    data=saved['replay']['buffer']
    states=torch.tensor(np.asarray([x[0] for x in data]),dtype=torch.float32)
    with torch.inference_mode():
        params=agent.param_net(states);q=agent.actor_net(states,params);actions=q.argmax(1)
        selected=params.gather(1,actions[:,None]).squeeze(1)
    visible=states[:,3]>0
    counts=lambda t:torch.bincount(t,minlength=7).tolist()
    report=dict(completed=saved['completed_episodes'],updates=saved['update_count'],replay=len(data),
        replay_state_greedy_actions=counts(actions),visible_state_greedy_actions=counts(actions[visible]),
        parameter_quantiles=torch.quantile(params.flatten(),torch.tensor([0.,.01,.5,.99,1.])).tolist(),
        selected_parameter_quantiles=torch.quantile(selected,torch.tensor([0.,.01,.5,.99,1.])).tolist(),
        outside_box_fraction=float(((selected<.3)|(selected>5)).float().mean()),
        greedy_q_quantiles=torch.quantile(q.max(1).values,torch.tensor([0.,.01,.5,.99,1.])).tolist(),
        warning='Offline forward over own training states only, no independent control test')
    print(json.dumps(report,indent=2))
    output=args.checkpoint.parent/'analysis';output.mkdir(exist_ok=True)
    (output/f"checkpoint-{saved['completed_episodes']:04d}-forward.json").write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
