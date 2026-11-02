"""Actual online tracker frames painted while the five-vehicle UE run is active."""
import speed_bootstrap
import argparse,shutil
from pathlib import Path
import live_crossroad_robust_tracker as previous
from online_result_recorder import Recorder
from calibrate import write as plain_write

runner=previous.runner
OUTPUT_ROOT=None
RECORDERS=[]

class Observation(previous.Observation):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.online=Recorder(OUTPUT_ROOT);RECORDERS.append(self.online)

    def observe(self,*args,**kwargs):
        state,row=super().observe(*args,**kwargs)
        self.online.publish(self.t.rgb,row,getattr(self.t,'frame',None),getattr(self.t,'runtime',None))
        return state,row

def materialize(folder,item):
    path=previous.materialize(folder,item)
    for n in ['live_genuine_online_results.py','online_result_recorder.py']:
        shutil.copy2(Path(__file__).parent/n,folder/'frozen-sources'/n)
    return path

def write(path,value):
    result=previous.write(path,value)
    if path.name=='configuration.json':
        value.update(true_online_output_recording=True,
            presented_video='online-output/true-online-output.mp4',
            online_overlay='Each actual completed tracker output is painted on thatsame pinnedinputRGB, onanindependent CPU writer duringlivecapture. No modelrerun, interpolation, associationcorrection or GT boxes.',
            online_video_cadence='True completed-output availability times, about8..10Hz; sensorcapturecadence separate. Afterrun onlycompresses alreadydrawnframes.')
        return plain_write(path,value)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--output',type=Path,required=True)
    args,_=parser.parse_known_args();OUTPUT_ROOT=args.output.resolve()
    runner.core.frontend=previous.previous.previous.previous.previous.frontend
    runner.cross.materialize=materialize;runner.Observation=Observation;runner.Detector=previous.Detector
    runner.write=write;runner.ACTION_FILTER=previous.action_filter
    success=False
    try:
        runner.main();success=True
    finally:
        for recorder in RECORDERS:recorder.close(encode=success)
