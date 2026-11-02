"""CPU-only Q/parameter diagnostic; never chooses or dispatches UE actions."""
import bootstrap
from pathlib import Path
import argparse,json
import numpy as np,torch
from author_aligned.source import create_agent
from author_aligned.run import NETWORKS

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--cases',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2);agent=create_agent(20261101,'cpu')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    for n in NETWORKS:getattr(agent,n).load_state_dict(saved[n])
    reports=[]
    for folder in args.cases:
        rows=[json.loads(x) for x in (folder/'steps.jsonl').read_text().splitlines()]
        state=torch.tensor(rows[-1]['next_state'],dtype=torch.float32).unsqueeze(0)
        parameters=agent.param_net(state).detach().requires_grad_(True)
        q=agent.actor_net(state,parameters)
        jacobian=torch.stack([torch.autograd.grad(q[0,k],parameters,retain_graph=True)[0][0] for k in range(7)])
        sweeps=[]
        with torch.inference_mode():
            for action in range(7):
                grid=torch.linspace(.3,5.,33);allp=parameters.detach().repeat(len(grid),1);allp[:,action]=grid
                values=agent.actor_net(state.repeat(len(grid),1),allp)
                best=int(values[:,action].argmax())
                sweeps.append(dict(action=action,network_parameter=float(parameters[0,action]),
                    q_at_network_parameter=float(q[0,action]),diagnostic_grid_argmax_parameter=float(grid[best]),
                    diagnostic_grid_max_q=float(values[best,action]),
                    stay_q_variation_when_other_duration_changed=float(values[:,0].max()-values[:,0].min())))
        reports.append(dict(folder=str(folder),state=state[0].tolist(),greedy_action=int(q.argmax(1)),
            parameters=parameters[0].tolist(),q=q[0].tolist(),
            own_action_gradients=jacobian.diagonal().tolist(),source_sum_of_all_q_gradients=jacobian.sum(0).tolist(),
            cross_q_parameter_jacobian=jacobian.tolist(),sweeps=sweeps))
    output=dict(scope='Read-only CPU diagnostics of trained critic and duration net; no simulator actions, no planner, no learned policy changes',
        checkpoint=str(args.checkpoint),reports=reports)
    args.output.write_text(json.dumps(output,indent=2),encoding='utf-8')
    for r in reports:
        print(r['folder'],'state',np.round(r['state'][:4],4),'greedy',r['greedy_action'])
        print('params',np.round(r['parameters'],3),'Q',np.round(r['q'],3))
        print('own dQ',np.round(r['own_action_gradients'],3),'source summed dQ',np.round(r['source_sum_of_all_q_gradients'],3))
        print('peraction grid max',[(s['action'],round(s['diagnostic_grid_argmax_parameter'],2),round(s['diagnostic_grid_max_q'],3)) for s in r['sweeps']])

if __name__=='__main__':main()
