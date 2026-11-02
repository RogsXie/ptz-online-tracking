"""Actual9m/s longroad, neural Q-selection with physical-limit action mask."""
import speed_bootstrap
import sys,json,numpy as np,torch
from pathlib import Path
import live_motion_tracking_9ms as fastscene
from motion_learning15 import LOW,HIGH,SECONDS

hierarchy=fastscene.hierarchy;latency=fastscene.latency;runner=fastscene.runner
CAMERA={}

class Observation(latency.Observation):
    def observe(self,*args,**kwargs):
        state,row=super().observe(*args,**kwargs)
        CAMERA.update(fov=float(self.t.diagnostics['fov']),pitch=float(self.t.diagnostics['rotation'][0]),yaw=float(self.t.diagnostics['rotation'][1]))
        return state,row

class Policy(hierarchy.Policy):
    def choose_action(self,state,train=True):
        action,param,params=super().choose_action(state,train=train);valid=np.ones(7,dtype=bool)
        fov=CAMERA['fov'];pitch=CAMERA['pitch'];yaw=CAMERA['yaw'];tol=1e-4
        if fov>=66.-tol:valid[6]=False
        if fov<=8.+tol:valid[5]=False
        if pitch>=-15.-tol:valid[2]=False
        if pitch<=-85.+tol:valid[1]=False
        if yaw>=65.-tol:valid[3]=False
        if yaw<=-65.+tol:valid[4]=False
        if valid[int(action)]:return action,param,params
        active=self.tracker if hierarchy.STAGE=='tracking' else self.coarse
        inputs=np.asarray(state if hierarchy.STAGE=='tracking' else state[:8],np.float32)
        with torch.inference_mode():
            q=active.actor_net(torch.tensor(inputs,device=active.device).unsqueeze(0),torch.tensor(params,dtype=torch.float32,device=active.device).unsqueeze(0))[0].cpu().numpy()
        masked=q.copy();masked[~valid]=-np.inf;selected=int(masked.argmax())
        lo,hi=(LOW[selected],HIGH[selected]) if hierarchy.STAGE=='tracking' else (.3,5.)
        raw=float(np.clip(params[selected],lo,hi))
        if not hasattr(self.tracker,'mask_stream'):
            folder=Path(sys.argv[sys.argv.index('--output')+1]);self.tracker.mask_stream=(folder/'physical-action-mask.jsonl').open('w',encoding='utf-8',buffering=1)
        self.tracker.mask_stream.write(json.dumps(dict(camera=CAMERA.copy(),stage=hierarchy.STAGE,original_action=int(action),selected_action=selected,valid=valid.tolist(),neural_q=q.tolist(),rule='Onlyphysicallyblockedmotionsexcluded;selectionstillneuralQargmax'))+'\n')
        return selected,np.array([raw],np.float32),params

def create_agent():
    hierarchy.policy=Policy();return hierarchy.policy

def write(path,value):
    if path.name=='configuration.json':
        value.update(physical_action_mask='Excludeonlyzoom beyondFOV8..66,pitch beyond-85..-15,yaw beyond-65..65;allremainingdirections chosenbyfrozenneuralQ. No geometriccentering actionrule.',
            mask_scope='Explicit execution constraint extension, not unmodifiedsourceargmax protocol')
        if value.get('all_learning_state_frozen') and hasattr(hierarchy.policy.tracker,'mask_stream'):hierarchy.policy.tracker.mask_stream.close()
    return fastscene.write(path,value)

if __name__=='__main__':
    runner.create_agent=create_agent;runner.resume=hierarchy.resume;runner.SpeedEnvironment=Observation
    runner.LOW=LOW;runner.HIGH=HIGH;runner.SECONDS_PER_RAW=SECONDS
    runner.layout=fastscene.scene.layout;runner.materialize=fastscene.materialize
    runner.AtomicTransport=hierarchy.Transport;runner.DisplayTransport=hierarchy.DisplayTransport
    runner.write=write;runner.main()
