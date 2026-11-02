"""Real UE I/O calibration using declared legal parameters; not RL training."""
import json
import warnings
import numpy as np
import torch
from author_aligned.environment import bind_environment, AtomicTransport
from author_aligned.source import REPRO, load_core, verify_source
from author_aligned.run import write_json
from integration.runtime import StandaloneArena
from integration.pretrained_detector import create_detector
from integration.experiment import make_extractor
from integration.transport import ACTION_RATES


def main():
    warnings.filterwarnings("ignore",category=SyntaxWarning)
    torch.set_num_threads(4)
    verify_source();_,pad,_=load_core()
    output=REPRO/"verification/author-aligned-20261031/real-ue-preflight"
    output.mkdir(parents=True,exist_ok=False)
    detector=create_detector(device="cuda");extractor=make_extractor("cuda");detector.warmup()
    rows=[]
    with StandaloneArena(output,headless=True) as arena:
        transport=AtomicTransport(arena.port,hud=False)
        try:
            env=bind_environment(transport,detector,extractor,"reject")
            env.io_bridge.scenario=REPRO/"scenarios/static_near.json"
            env.io_bridge.target_count=1
            with (output/"rows.jsonl").open("w",encoding="utf-8",buffering=1) as log:
                for action in range(7):
                    for parameter in (.3,2.,5.):
                        env.io_bridge.seed=900000+len(rows)
                        state=env.reset();before=transport.diagnostics.copy()
                        nxt,reward,done,_=env.step(pad(action,np.array([parameter])))
                        after=transport.diagnostics
                        pan,tilt,zoom=ACTION_RATES[action];duration=parameter/5
                        expected=[np.clip(before["rotation"][0]+tilt*25*duration,-25,25),
                            np.clip(before["rotation"][1]+pan*25*duration,-40,40),
                            np.clip(before["fov"]-zoom*5*duration,4.4085,60)]
                        actual=[after["rotation"][0],after["rotation"][1],after["fov"]]
                        row={"case":len(rows)+1,"action":action,"raw":parameter,
                            "state":list(map(float,state)),"next_state":list(map(float,nxt)),
                            "reward":float(reward),"duration":duration,
                            "actual":actual,"expected":list(map(float,expected)),
                            "max_pose_error":float(np.max(np.abs(np.array(actual)-expected))),
                            "time_error":abs(after["sim_time"]-before["sim_time"]-duration),
                            "no_gt_follow":not after["demo_geometry_follow"] and not after["free_preview"]}
                        rows.append(row);log.write(json.dumps(row)+"\n")
                        print(f"UE_CALIBRATION {len(rows)}/21 action={action} raw={parameter} error={row['max_pose_error']:.7f}",flush=True)
        finally: transport.close()
    result={"passed":all(r["max_pose_error"]<1e-5 and r["time_error"]<1e-5 and r["no_gt_follow"] for r in rows),
        "cases":len(rows),"rows":rows,"training_updates":0,"detector":detector.provenance,
        "reid":extractor.metadata(),"note":"Legal-parameter driver calibration only; not learned policy performance"}
    write_json(output/"results.json",result)
    print(json.dumps({"passed":result["passed"],"cases":len(rows),"output":str(output)}))
    if not result["passed"]: raise SystemExit(1)


if __name__=="__main__": main()
