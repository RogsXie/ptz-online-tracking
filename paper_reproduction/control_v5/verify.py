"""Regression checks for timeout bootstrapping and clock-free observations."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import numpy as np
import torch
from control_v4.environment import StaticEnvironment
from control_v4.agent import StaticAgent
from control_v5.environment import ContinuingState,ContinuingEnvironment,transition_terminal
from control_v5.agent import ContinuingAgent
from control_v3.run import frozen,network_hashes,write_json
from integration.runtime import REPRO


def constant_q_agent(device,terminal):
    agent=ContinuingAgent(seed=92,device=device)
    with torch.no_grad():
        for net in (agent.actor_net,agent.actor_target):
            for p in net.parameters():p.zero_()
            net.base.layers[-1].bias.fill_(2.)
    parameters=np.full(7,.145,np.float32)
    s=np.array([.04,0.,-.5,.9,0.,0.,0.,0.],np.float32)
    ns=s.copy();ns[0]=.02
    for _ in range(256):agent.observe(s,(0,.145),0.,ns,terminal,parameters)
    return agent


def main():
    torch.set_num_threads(2)
    checks=[]
    detection=SimpleNamespace(center=(.51,.50),area=.005,confidence=.92)
    early,late=ContinuingState(10),ContinuingState(10)
    a=early.build(detection,(1,.1));b=late.build(detection,(4,9.9))
    assert np.array_equal(a,b) and np.array_equal(a[4:],np.zeros(4))
    loss=late.build(None,(0,.1))
    assert np.array_equal(loss[:3],b[:3]) and loss[3]==0
    checks.append({"check":"current-image state independent of countdown and prior actions; loss explicit","passed":True})
    ns=np.arange(8,dtype=np.float32)
    with patch.object(StaticEnvironment,"step",return_value=(ns,.2,True,{})):
        env=ContinuingEnvironment.__new__(ContinuingEnvironment)
        actual_next,reward,collector_done,row=env.step(0,.1)
    assert actual_next is ns and collector_done and row["truncated"] and not row["terminated"]
    assert transition_terminal(row) is False
    assert transition_terminal({"terminated":True,"truncated":False}) is True
    try:transition_terminal({"done":True})
    except ValueError:pass
    else:raise AssertionError("Ambiguous old done accepted")
    checks.append({"check":"collector stops but learner bootstraps exact final RGB observation; explicit genuine termination supported","passed":True})
    for device in ("cpu","cuda"):
        continuing=constant_q_agent(device,transition_terminal(row))
        absorbing=constant_q_agent(device,True)
        metrics=continuing.update(); terminal_metrics=absorbing.update()
        expected=.5*(2.-.98*2.)**2
        assert abs(metrics["q_loss"]-expected)<1e-7,(metrics,expected)
        assert abs(terminal_metrics["q_loss"]-1.5)<1e-7
        checks.append({"device":device,"check":"analytical Bellman regression: timeout retains gamma*Q; actual terminal masks it",
            "passed":True,"timeout_q_loss":metrics["q_loss"],"expected":expected,"terminal_q_loss":terminal_metrics["q_loss"]})
        receipt={}
        with frozen(continuing,receipt):continuing.select_action(a,False)
        assert receipt["networks_unchanged"] and receipt["sampling_rng_unchanged"] and receipt["counters_unchanged"]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"checkpoint.pt"
            continuing.save(path)
            restored=ContinuingAgent.load(path,device=device)
            assert continuing.update()==restored.update()
            assert network_hashes(continuing)==network_hashes(restored)
            StaticAgent(seed=92,device=device).save(path)
            try:ContinuingAgent.load(path,device=device)
            except ValueError:pass
            else:raise AssertionError("Finite-task checkpoint mixed into continuing task")
        checks.append({"device":device,"check":"frozen inference and exact restore; incompatible finite-task weights rejected","passed":True})
    output=REPRO/"verification/continuing-static-v5-20261031/contract-checks.json"
    write_json(output,{"passed":True,"checks":checks,"scope":"implementation/target regression; not live control qualification"})
    print({"passed":True,"checks":len(checks),"output":str(output)})


if __name__=="__main__":main()
