"""Fresh frozen control on the32declared development-only true corners.

Only the evaluation layout factory differs from evaluate_audited. The original
P-DQN controller, observations, rewards, physical mapping and acceptance stay
identical. Seeds33020000..33020031 are development data, never final tests.
"""
import bootstrap
import sys
import evaluate_audited as runner
from evaluation_scenes import calibration_corner

def corner_layout(split,index):
    if split!='development' or not 0<=index<32:
        raise ValueError('This entry is exclusively32development corner cases')
    item=calibration_corner(index)
    item['source_case_index']=item['index']
    item['index']=index
    item['layout_profile']='development_true_corners_32'
    return item

if __name__=='__main__':
    if '--count' not in sys.argv:sys.argv.extend(['--count','32'])
    if '--split' not in sys.argv:sys.argv.extend(['--split','development'])
    runner.layout=corner_layout
    runner.main()
