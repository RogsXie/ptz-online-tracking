"""Focused safety/transition checks for first-lock scale acquisition."""
import speed_bootstrap
from types import SimpleNamespace
import numpy as np
import torch
from stable_locked_control import filter_action,STARTUP_AREA

def fixture(area=.0015,x=0.,generation=1):
    width=height=area**.5
    row=dict(visible=True,area=area,sim_time=0.,target_label='car',
        bbox=[.5+x/2-width/2,.5-height/2,.5+x/2+width/2,.5+height/2])
    diag=dict(fov=66.,rotation=[-42.,10.,0.])
    env=SimpleNamespace(latest_rgb_observation=row,t=SimpleNamespace(diagnostics=diag),
        cfff=SimpleNamespace(info={'target_generation':generation}),
        camera_limits=dict(fov_min_deg=6.,fov_max_deg=66.,pan_deg_s=180.,tilt_deg_s=100.,
            zoom_fov_deg_s=100.,pitch_limits_deg=[-85.,-15.],yaw_limits_deg=[-65.,65.]))
    agent=SimpleNamespace(device='cpu',actor_net=lambda *_:torch.zeros((1,7)))
    state=np.zeros(15,np.float32);state[0]=x
    return env,agent,state,np.ones(7,np.float32)

def run(env,agent,state,params):return filter_action(env,agent,state,0,.01,params)

def main():
    env,agent,state,params=fixture(x=.05) #24px: moving target may zoom before exact centring.
    action,duration,info=run(env,agent,state,params)
    assert action==5 and info['reason']=='locked_startup_zoom_in'
    assert 0<duration<=.06 and env.t.zoom_gain_override==1.
    assert info['desired_fov_deg']<66. and info['phase']=='acquire'
    # A 96px error must still be corrected before zooming.
    env,agent,state,params=fixture(x=.2)
    action,_,info=run(env,agent,state,params)
    assert action in [1,2,3,4] and info['phase']=='acquire'
    # New target identity restarts scale acquisition, while a completed lock
    # does not repeatedly restart when subsequent visible area fluctuates.
    env,agent,state,params=fixture(area=STARTUP_AREA)
    assert run(env,agent,state,params)[0]==0 and not env.startup_scale_pending
    env.latest_rgb_observation['area']=.0015
    assert run(env,agent,state,params)[0]==0
    env.cfff.info['target_generation']=2
    assert run(env,agent,state,params)[0]==5
    # Area limit takes priority over startup; minimum FOV cannot be exceeded.
    env,agent,state,params=fixture(area=.025);env.t.diagnostics['fov']=30.
    assert run(env,agent,state,params)[0]==6
    env,agent,state,params=fixture();env.t.diagnostics['fov']=6.
    assert run(env,agent,state,params)[0]!=5
    # Genuine loss still clears the box and expands view; never startup-zooms.
    env,agent,state,params=fixture();env.t.diagnostics['fov']=30.
    env.latest_rgb_observation.update(visible=False,area=None,bbox=None)
    assert run(env,agent,state,params)[0]==6
    print('PASS: first-lock active zoom, rough centring gate, bounded pulse, per-identity completion, area/FOV and loss guards')

if __name__=='__main__':main()
