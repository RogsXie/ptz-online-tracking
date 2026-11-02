"""Regress axis starvation through the real action filter and its limits."""
import speed_bootstrap
from types import SimpleNamespace
import numpy as np,torch
import precision_policy_v2 as precision
import stable_locked_control as control
from camera_centering_v4 import preferred_axis,pulse_error


def case(x,y,area=.016,enabled=True,pitch=-42.):
    control.core.LOW,control.core.HIGH=precision.LOW,precision.HIGH
    row=dict(visible=True,x=x/480,y=y/270,area=area,sim_time=1.,bbox=[.45,.45,.55,.55],
        measured_perception_age_s=.09,target_label='car')
    limits=dict(fov_min_deg=6.,fov_max_deg=66.,zoom_fov_deg_s=100.,pan_deg_s=180.,tilt_deg_s=100.,
        yaw_limits_deg=[-65.,65.],pitch_limits_deg=[-85.,-5.])
    env=SimpleNamespace(latest_rgb_observation=row,t=SimpleNamespace(diagnostics=dict(fov=21.,rotation=[pitch,10.,0.])),
        camera_limits=limits,cfff=SimpleNamespace(info=dict(target_generation=1,locked_bearing_velocity_deg_s=[0.,0.])))
    q=lambda *args:torch.tensor([[0.,2.,1.,10.,1.,0.,0.]])
    agent=SimpleNamespace(precision_control_v3=True,precision_centering_v4=enabled,
        actor_net=q,device='cpu',precision_motion_budget=True)
    state=np.zeros(15,np.float32);state[:2]=[x/480,y/270]
    params=np.array([1.,4.,4.,4.,4.,1.,1.],np.float32)
    return env,agent,state,params


def run(x,y,**kwargs):
    env,a,state,p=case(x,y,**kwargs)
    return control.filter_action(env,a,state,3,.017,p)


def main():
    assert run(30.,46.,enabled=False)[0]==3
    result=run(30.,46.);assert result[0]==1 and result[2]['centering_priority_axis']=='tilt'
    assert .001<=result[1]<=.1
    assert run(46.,30.)[0]==3
    # Mechanical limits still dominate a large error on an unreachable axis.
    assert run(30.,46.,pitch=-85.)[0] not in (1,2)
    # A target 20px off center must be corrected before further magnification.
    assert run(20.,0.,area=.002,enabled=False)[0]==5
    assert run(20.,0.,area=.002)[0]==3
    # The existing 2% maximum area guard cannot be suppressed by centering.
    assert run(100.,100.,area=.03)[0]==6
    acquisition=run(200.,0.,area=.002)
    assert acquisition[0]==3 and .017<acquisition[1]<=.1
    valid=np.ones(7,bool)
    assert preferred_axis([30.,12.],valid,0,{0:.39,1:0.},.4)==1
    assert preferred_axis([20.,19.],valid,0)==1
    lead=pulse_error(np.array([10.,-20.]),np.array([1000.,-1000.]),.2)
    assert np.allclose(lead,[15.,-30.])
    assert np.allclose(pulse_error(np.array([10.,-20.]),np.array([-1000.,1000.]),.2),[10.,-20.])
    print('PASS: starvation reproduced and fixed; primary-axis priority; mechanical/pulse/area limits; tighter prezoom centering')


if __name__=='__main__':main()
