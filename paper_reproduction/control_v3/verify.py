"""Metamorphic learning/stop-contract checks before any v3 live training."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
import torch
from control_v3.agent import ImprovedAgent
from control_v3.run import frozen, network_hashes, review_probe, write_json
from control_learning.environment import ControlState
from integration.runtime import REPRO


def main():
    torch.set_num_threads(4)
    results = []
    for device in ("cpu", "cuda"):
        agent = ImprovedAgent(seed=6431, device=device)
        states = torch.randn(3,8,device=device)
        params = torch.full((3,7),.145,device=device,requires_grad=True)
        values = agent.actor_net(states,params)
        largest_cross = 0.
        for action in range(7):
            gradient, = torch.autograd.grad(values[:,action].sum(),params,retain_graph=True)
            inactive = [other for other in range(7) if other!=action]
            largest_cross = max(largest_cross, float(gradient[:,inactive].abs().max()))
            altered = params.detach().clone()
            altered[:,inactive] = .249
            assert torch.equal(values[:,action],agent.actor_net(states,altered)[:,action])
        assert largest_cross == 0.
        total_gradient, = torch.autograd.grad(values[:,:5].sum(),params,retain_graph=True)
        own_gradient, = torch.autograd.grad(values[:,2].sum(),params)
        assert torch.equal(total_gradient[:,2],own_gradient[:,2])
        results.append({"device":device,"check":"unexecuted-duration perturbation and cross-gradient isolation","passed":True,"largest_cross_gradient":largest_cross})
        assert agent.update() is None
        rng = np.random.default_rng(5)
        for index in range(300):
            s = rng.uniform(-1,1,8).astype(np.float32)
            next_s = rng.uniform(-1,1,8).astype(np.float32)
            parameters = rng.uniform(.04,.25,7).astype(np.float32)
            action = index%5
            agent.observe(s,(action,float(parameters[action])),float(rng.normal()),next_s,index%70==0,parameters)
        hashes = network_hashes(agent)
        for _ in range(100):
            metrics = agent.update()
            assert np.isfinite(list(value for value in metrics.values() if isinstance(value,(int,float)))).all()
        assert metrics["target_sync"] and hashes["actor_net"]!=network_hashes(agent)["actor_net"]
        assert hashes["param_net"]!=network_hashes(agent)["param_net"]
        assert all(torch.equal(a,b) for a,b in zip(agent.actor_net.parameters(),agent.actor_target.parameters()))
        results.append({"device":device,"check":"100 real synthetic-data optimizer steps and exact target sync","passed":True})
        receipt = {}
        with frozen(agent,receipt):
            for _ in range(8):
                action,duration=agent.select_action(np.zeros(8,np.float32),explore=False)
                assert action in range(5) and .04<=duration<=.25
                assert all(not getattr(agent,name).training for name in ("actor_net","actor_target","param_net","param_net_target"))
        assert receipt["networks_unchanged"] and receipt["sampling_rng_unchanged"] and receipt["counters_unchanged"]
        results.append({"device":device,"check":"four-network frozen greedy inference does not change RNG/replay/learning","passed":True})
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"resume.pt"
            agent.save(path,{"synthetic_verification_only":True})
            restored=ImprovedAgent.load(path,device=device)
            for _ in range(10):
                assert agent.select_action(np.zeros(8,np.float32))==restored.select_action(np.zeros(8,np.float32))
            assert agent.update()==restored.update()
            assert network_hashes(agent)==network_hashes(restored)
        results.append({"device":device,"check":"save/reload reproduces exploration, replay sampling and subsequent gradient update","passed":True})
        before_count=agent.update_count
        with torch.no_grad():
            agent.param_net.layers[-1].bias[0]=float("nan")
        try:
            agent.select_action(np.zeros(8,np.float32))
        except FloatingPointError:
            pass
        else:
            raise AssertionError("NaN duration was not rejected before execution")
        assert agent.update_count==before_count
        results.append({"device":device,"check":"nonfinite proposals stop before actuator dispatch","passed":True})
    builder=ControlState()
    detection=SimpleNamespace(center=(.6,.4),area=.01,confidence=.95)
    state=builder.build(detection,(4,.12))
    assert np.isclose(state[4],4/6) and np.isclose(state[5],.12/.25)
    lost=builder.build(None,(1,.08))
    assert lost[3]==0 and np.array_equal(lost[:3],state[:3]) and np.isclose(lost[4],1/6)
    results.append({"check":"latest executed action/duration history and explicit RGB-loss state","passed":True})
    probe={"collapsed_cases":4,"lost_cases":0,"reversal_mean":0,"passed_cases":0,"total":8,"reward_mean":2.,"center_mean":.1}
    assert not review_probe(probe,None,8) and review_probe(probe,None,16)
    probe.update(collapsed_cases=0,passed_cases=7,reversal_mean=14)
    assert review_probe(probe,dict(probe),16)
    probe.update(reversal_mean=1)
    assert review_probe(probe,None,32)
    results.append({"check":"expected early exploration tolerated; persistent collapse/oscillation and bounded budget stop","passed":True})
    output=REPRO/"verification/improved-control-v3-20261031/contract-checks.json"
    write_json(output,{"passed":True,"checks":results,"scope":"code/optimizer contract tests, not live tracking success"})
    print(json.dumps({"passed":True,"checks":len(results),"output":str(output)}))


if __name__=="__main__":
    main()
