"""Observation/compatibility checks plus unchanged-learner comparison."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import numpy as np
import torch
from control_v3.agent import ImprovedAgent
from control_v3.run import network_hashes, write_json, frozen
from control_v4.agent import StaticAgent
from control_v4.environment import StaticState, StaticEnvironment
from integration.runtime import REPRO


def main():
    torch.set_num_threads(2)
    checks=[]
    target=SimpleNamespace(center=(.508,.494),area=.0045,confidence=.91)
    old=SimpleNamespace(center=(.52,.48),area=.0046,confidence=.90)
    a,b=StaticState(10),StaticState(10)
    a.build(old,(3,.1));sa=a.build(target,(4,.1))
    sb=b.build(target,(0,.2))
    assert np.array_equal(sa,sb)
    assert np.isclose(sa[4],.98) and np.array_equal(sa[5:],np.zeros(3))
    later=b.build(target,(0,.25))
    assert np.array_equal(sa[:4],later[:4]) and later[4]<sa[4]
    lost=b.build(None,(1,.04))
    assert np.array_equal(lost[:3],later[:3]) and lost[3]==0
    b.build(None,(0,20.))
    assert b.build(None)[4]==0
    checks.append({"check":"same current image/time is invariant to different executed histories; finite horizon and loss explicit","passed":True})
    try:
        StaticEnvironment(None,None,None,{"id":"uniform_motion"})
    except ValueError: pass
    else: raise AssertionError("Static state silently used for moving target")
    checks.append({"check":"motion stages blocked","passed":True})
    for device in ("cpu","cuda"):
        static=StaticAgent(seed=776,device=device)
        previous=ImprovedAgent(seed=776,device=device)
        assert network_hashes(static)==network_hashes(previous)
        changed={k for k,v in asdict(static.config).items() if v!=asdict(previous.config)[k]}
        assert changed=={"reward_mode","max_steps"}
        rng=np.random.default_rng(6)
        for index in range(280):
            state=np.r_[rng.uniform(-1,1,4),rng.uniform(0,1),[0,0,0]].astype(np.float32)
            next_state=state.copy();next_state[4]=max(0.,state[4]-.02)
            parameters=rng.uniform(.04,.25,7).astype(np.float32)
            action=index%5; reward=float(rng.normal())
            for agent in (static,previous):
                agent.observe(state,(action,float(parameters[action])),reward,next_state,index%60==0,parameters)
        for _ in range(20):
            assert static.update()==previous.update()
        assert network_hashes(static)==network_hashes(previous)
        checks.append({"check":"same input replay produces bit-identical v3/v4 learning updates; only observation semantics change","device":device,"passed":True})
        receipt={}
        with frozen(static,receipt):
            action=static.select_action(sa,False)
            assert action[0] in (0,1,2,3,4)
        assert receipt["networks_unchanged"] and receipt["sampling_rng_unchanged"] and receipt["counters_unchanged"]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"new.pt"
            static.save(path,{"state_features":"static_v4_verification"})
            restored=StaticAgent.load(path,device=device)
            assert static.update()==restored.update()
            assert network_hashes(static)==network_hashes(restored)
            previous.save(path)
            try: StaticAgent.load(path,device=device)
            except ValueError: pass
            else: raise AssertionError("v3 weights accepted as the new observation contract")
        checks.append({"check":"frozen inference, exact reload and incompatible v3 checkpoint rejection","device":device,"passed":True})
    output=REPRO/"verification/static-observation-v4-20261031/contract-checks.json"
    write_json(output,{"passed":True,"checks":checks,"scope":"code-contract validation, not live control qualification"})
    print({"passed":True,"checks":len(checks),"output":str(output)})


if __name__=="__main__":main()
