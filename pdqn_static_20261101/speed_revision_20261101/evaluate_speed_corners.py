"""Same32known development corners; no new heldout cases are selected here."""
import speed_bootstrap
import sys
import speed_runner
import evaluate_speed
from evaluation_scenes import calibration_corner

def corners(split,index):
    if split!='development' or not 0<=index<32:raise ValueError('Known32corner fixture only')
    item=calibration_corner(index);item['source_case_index']=item['index'];item['index']=index
    return item

if __name__=='__main__':
    speed_runner.layout=corners
    if '--count' not in sys.argv:sys.argv.extend(['--count','32'])
    evaluate_speed.main()
