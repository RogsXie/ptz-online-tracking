"""New dynamic-trained sourceP-DQN on exactly the original motion/stop scene."""
import speed_bootstrap
import live_speed_v5 as runner
import live_uniform_motion_stop as scene
import live_fast_tiny_corner as fast
import motion_precision_v2 as actuator
from motion_learning15 import create_agent,resume,ObservationEnvironment,LOW,HIGH,SECONDS,PROFILE

def write(path,value):
    if path.name=='configuration.json':
        value.update(execution_profile=PROFILE,rates=list(fast.RATES),motion=scene.MOTION,
            execution_calibration='Dynamic-trained15input sourceP-DQN; exclusive7actions, angularraw.25..10, opticsgain2 andshortStay. Networkchoosesall directions/rawdurations.',
            observation_extension='Original8features plusRGB+owncamera derived angularvelocity2,FOV,feedbackdt,owncamera increments3. No GT,trajectory,knownvelocityorstop schedule in control.',
            qualification='Fresh same-scene dynamic development comparison; not statisticalindependentqualification',
            scope='Same native1.5m/s18s thenstop, sameinitialtinycorner/camera/road; actualrealtime RGBand originalwalltimestamps')
    return fast.original_write(path,value)

if __name__=='__main__':
    runner.create_agent=create_agent;runner.resume=resume
    runner.SpeedEnvironment=ObservationEnvironment
    runner.LOW=LOW;runner.HIGH=HIGH;runner.SECONDS_PER_RAW=SECONDS
    runner.layout=scene.layout;runner.materialize=scene.materialize
    runner.AtomicTransport=actuator.OpticsTransport;runner.DisplayTransport=actuator.OpticsDisplayTransport
    runner.write=write;runner.main()
