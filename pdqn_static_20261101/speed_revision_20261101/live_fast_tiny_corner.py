"""Fast tiny-corner development demo; learned exclusive directions unchanged.

Explicit execution extension: real pan180/tilt100/zoom100 rates, reachable
pitch range, and monotonic stretching of larger learned duration parameters.
Small raw parameters<=1 retain the previous per-command displacement. No
coordinates, GT directions, simultaneous actions, or trajectory replay.
"""
import speed_bootstrap
import json,numpy as np
import live_speed_v5 as runner
import live_tiny_corner_zoom15 as tiny
from scenes import configuration,INIT,ground_position

RATES=(180.,100.,100.)
original_write=runner.write
original_resume=runner.resume

def executed_raw(raw):
    raw=float(np.clip(raw,.3,5.))
    return raw*(1.+max(0.,raw-1.)/4.)

def resume(agent,path):
    saved=original_resume(agent,path)
    choose=agent.choose_action
    def calibrated_choose(state,train=True):
        action,param,parameters=choose(state,train=train)
        # The unchanged source selects all directions and durations. Only the
        # execution-facing selected parameter is rescaled; original outputs
        # remain in all_parameters for disclosure and reproduction.
        if int(action)==0:return action,param,parameters
        return action,np.array([executed_raw(float(param[0]))],np.float32),parameters
    agent.choose_action=calibrated_choose
    return saved

def materialize(folder,item):
    folder.mkdir(parents=True,exist_ok=True)
    cfg=configuration(item,RATES)
    init=dict(INIT,camera_location_cm=[-2200.,0.,item['camera_height_cm']])
    cfg['camera'].update(location_cm=init['camera_location_cm'],pitch_limits_deg=[-85.,-30.])
    cfg['targets'][0]['path']['origin_cm']=ground_position(*item['planned_center'],item['fov_deg'],init)
    cfg['scope_note']='Same tinyupperleft initialview; pitchlimit raised from-40to-30 because required target direction was outside former reach. Explicit faster execution calibration, not originalpaperprotocol.'
    path=folder/'scene-fast-tiny-upper-left.json';path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    return path

def write(path,value):
    if path.name=='configuration.json':
        value.update(execution_profile='fast_tiny_pan180_tilt100_zoom100_monotone_duration_demo',rates=list(RATES),
            raw_highs=[5.,10.,10.,10.,10.,10.,10.],
            execution_calibration='Originalsource choosesaction/raw; nonStay executedraw=clip(raw,.3,5)*(1+max(0,clip(raw,.3,5)-1)/4). Learnedraw<=1 displacementpreserved; largestcommand2x. all_parameterslogs unchanged originaloutputs.',
            qualification='Fresh development video only; no resumedvalidation or learnednetwork updates',
            scope='Per-action executionmapping and physicalrates changed; exclusive7actions retained. No handalignmentrule or parallelmotion.')
    return original_write(path,value)

if __name__=='__main__':
    runner.layout=tiny.layout
    runner.materialize=materialize
    runner.resume=resume
    runner.SECONDS_PER_RAW=(.1,.4/100.,.4/100.,.8/180.,.8/180.,1./100.,1./100.)
    runner.write=write
    runner.main()
