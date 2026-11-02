"""Frozen static+learned motion observation adapter, same original moving scene."""
import speed_bootstrap
import live_speed_v5 as runner
import live_uniform_motion_stop as scene
import live_fast_tiny_corner as fast
import motion_precision_v2 as actuator
from motion_learning15 import LOW,HIGH,SECONDS
from motion_adapter15 import create_agent,resume,ObservationEnvironment,PROFILE

def write(path,value):
    if path.name=='configuration.json':
        value.update(execution_profile=PROFILE,rates=list(fast.RATES),motion=scene.MOTION,
            execution_calibration='Samegain2 optics andshortStay; directions/rawdurations chosenby sourceP-DQN. Oldstatic networkparametersfrozen;new motioninputcolumns learned withoriginalRLupdates.',
            observation_extension='OnlyRGB+owncamera derivedtargetbearingvelocityadapter;noGT,knownpath,velocityorstop schedule supplied to control.',
            qualification='Same-scene freshrealtime development comparison; not statisticalqualification',scope='Same1.5m/s18s thenstop tinyinitialview/road;exclusive7actions retained')
    return fast.original_write(path,value)

if __name__=='__main__':
    runner.create_agent=create_agent;runner.resume=resume;runner.SpeedEnvironment=ObservationEnvironment
    runner.LOW=LOW;runner.HIGH=HIGH;runner.SECONDS_PER_RAW=SECONDS
    runner.layout=scene.layout;runner.materialize=scene.materialize
    runner.AtomicTransport=actuator.OpticsTransport;runner.DisplayTransport=actuator.OpticsDisplayTransport
    runner.write=write;runner.main()
