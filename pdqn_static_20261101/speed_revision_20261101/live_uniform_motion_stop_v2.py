"""Same uniform1.5m/s stop scenario, corrected highzoom duty and feedback."""
import speed_bootstrap
import motion_precision_v2 as actuator
import live_uniform_motion_stop as scene
import live_fast_tiny_corner as fast

runner=actuator.runner

def write(path,value):
    if path.name=='configuration.json':
        value.update(execution_profile='uniform_motion_stop_angular_gain2_short_stay',rates=list(fast.RATES),motion=scene.MOTION,
            execution_calibration='Frozen P-DQN directions/rawduration; angulargain=min(1,2*(tan(FOV/2)/tan(33deg))**2), zoomgain=min(1,FOV/66). Staysecondsperraw.01 (formerly.1), reducing blindstationarywait. Allactualcommands logged.',
            actuator_inputs='CameraFOVonly for opticscalibration; no targeterror/velocity/GT or motionschedule in control.',
            qualification='Same-origin dynamic development comparison, frozennetworks, no fullqualification restart',
            scope='Exactsame native1.5m/s straightmotion18s thenstop, same camera/initialview/road; exclusive7actions retained')
    return fast.original_write(path,value)

if __name__=='__main__':
    runner.layout=scene.layout
    runner.materialize=scene.materialize
    runner.resume=fast.original_resume
    runner.AtomicTransport=actuator.OpticsTransport
    runner.DisplayTransport=actuator.OpticsDisplayTransport
    runner.SECONDS_PER_RAW=(.01,.4/100.,.4/100.,.8/180.,.8/180.,1./100.,1./100.)
    runner.write=write
    runner.main()
