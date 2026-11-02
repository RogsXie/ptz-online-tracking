"""Read-only CPU reproduction of frozen decisions and physical action effects.

The unedited pinned networks are loaded to reproduce outputs. This program
never calls choose_action, updates a learner, or sends simulator commands.
"""
import bootstrap
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from author_aligned.source import create_agent,verify_source

RATES=((0,0,0),(0,-1,0),(0,1,0),(1,0,0),(-1,0,0),(0,0,1),(0,0,-1))

def lines(path):return [json.loads(s) for s in path.read_text().splitlines()]
def clamp(value,lo,hi):return min(hi,max(lo,value))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--folders',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();assert not args.output.exists();torch.set_num_threads(2)
    digest=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest();saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    agent=create_agent(20261101,'cpu')
    for n in ('actor_net','param_net'):getattr(agent,n).load_state_dict(saved[n])
    records=[];states=[];logged_parameters=[];logged_actions=[];physical_errors=[];counts=[]
    for folder in args.folders:
        report=json.loads((folder/'independent-results.json').read_text())
        assert report['configuration']['checkpoint_sha256']==digest
        assert len(report['episodes'])==report['configuration']['count']
        assert json.loads((folder/'frozen-verification.json').read_text())['networks_targets_optimizers_replay_rng_unchanged']
        cases=sorted(folder.glob('case-*/steps.jsonl'));assert len(cases)==len(report['episodes'])
        count=0
        for path in cases:
            rows=lines(path);diagnostics=lines(path.with_name('offline-diagnostics.jsonl'))
            assert len(rows)==len(diagnostics)
            assert np.max(np.abs(np.array(diagnostics[0]['rotation'])-[-60.,0.,0.]))<1e-6
            for row,before,after in zip(rows[1:],diagnostics,diagnostics[1:]):
                action=row['action'];parameter=row['raw_parameter'];dt=clamp(parameter,.3,5.)/5.
                assert abs(row['duration']-dt)<1e-10
                assert parameter==row['all_parameters'][action]
                assert len(row['input_state'])==8
                assert abs(row['sim_time']-after['sim_time'])<1e-6
                assert abs(after['sim_time']-before['sim_time']-dt)<1e-5
                pan,tilt,zoom=RATES[action]
                expected=[clamp(before['rotation'][0]+tilt*2.*dt,-80.,-40.),
                    clamp(before['rotation'][1]+pan*4.*dt,-65.,65.),before['rotation'][2],
                    clamp(before['fov']-zoom*5.*dt,14.,66.)]
                actual=[*after['rotation'],after['fov']]
                physical_errors.append(float(np.max(np.abs(np.array(actual)-expected))))
                states.append(row['input_state']);logged_parameters.append(row['all_parameters']);logged_actions.append(action)
                records.append(dict(action=action,duration=dt,raw_parameter=parameter))
                count+=1
        counts.append(dict(folder=str(folder),episodes=len(cases),decisions=count))
    states=torch.tensor(states,dtype=torch.float32);expected=torch.tensor(logged_parameters,dtype=torch.float32)
    all_parameters=[];all_actions=[];margins=[]
    with torch.inference_mode():
        for batch in states.split(256):
            params=agent.param_net(batch);q=agent.actor_net(batch,params)
            all_parameters.append(params);all_actions.append(q.argmax(1))
            top=q.topk(2,dim=1).values;margins.extend((top[:,0]-top[:,1]).tolist())
    params=torch.cat(all_parameters);actions=torch.cat(all_actions)
    parameter_error=float((params-expected).abs().max())
    mismatch=int((actions!=torch.tensor(logged_actions)).sum())
    physical_error=max(physical_errors)
    checks=dict(pinned_author_source_verified=bool(verify_source()),all_discrete_actions_match_original_networks=mismatch==0,
        all_seven_continuous_outputs_match_cpu_reproduction=parameter_error<3e-4,
        selected_parameter_and_duration_exact=True,all_ue_camera_changes_match_action_rate_duration_and_limits=physical_error<2e-5)
    durations={}
    for action in range(7):
        values=[r['duration'] for r in records if r['action']==action]
        if values:durations[action]=dict(count=len(values),min=min(values),median=float(np.median(values)),max=max(values),std=float(np.std(values)))
    report=dict(passed=all(checks.values()),checks=checks,checkpoint=str(args.checkpoint),checkpoint_sha256=digest,
        scope='Independent CPU output and recorded physical-effect audit; no simulator commands, training or hand action rules',
        folders=counts,total_decisions=len(records),discrete_action_mismatches=mismatch,
        continuous_parameter_max_abs_cpu_gpu_difference=parameter_error,min_greedy_q_margin=min(margins),
        max_physical_pose_fov_error_deg=physical_error,actual_duration_seconds_by_action=durations,
        clipped_execution_count=sum(not .3<=r['raw_parameter']<=5. for r in records))
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
    assert report['passed'],'Recorded decisions or physical effects did not reproduce; inspect report'

if __name__=='__main__':main()
