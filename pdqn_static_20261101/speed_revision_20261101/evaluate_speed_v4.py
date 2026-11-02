"""Frozen v3or released-model campaign on exactly paired initial layouts."""
import speed_bootstrap
from speed_bootstrap import configure_precision_project
import argparse,json,hashlib,warnings
from pathlib import Path
import torch
import train as legacy
from speed_runner_v4 import create_agent,resume,campaign,frontend,AuditedSpeed,AuditedOld
from author_aligned.source import create_agent as old_agent
from author_aligned.environment import AtomicTransport
from integration.runtime import StandaloneArena
from evaluation_scenes import calibration_corner

def corners(split,index):
    assert split=='development' and 0<=index<32
    item=calibration_corner(index);item['source_case_index']=item['index'];item['index']=index
    return item

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=80);p.add_argument('--split',choices=['development','validation','final_test'],default='development')
    p.add_argument('--old-baseline',action='store_true');p.add_argument('--corners',action='store_true');p.add_argument('--validation-report',type=Path);args=p.parse_args()
    if args.split=='final_test' and not args.old_baseline:
        assert args.validation_report
        report=json.loads(args.validation_report.read_text(encoding='utf-8'))
        assert report['qualification']['qualified'] and report['qualification']['count']==160
        assert report['configuration']['checkpoint_sha256']==hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        assert json.loads((args.validation_report.parent/'frozen-verification.json').read_text())['networks_targets_optimizers_replay_rng_unchanged']
    legacy.exclusive();configure_precision_project();warnings.filterwarnings('ignore');torch.set_num_threads(4)
    detector,extractor=frontend()
    if args.old_baseline:
        a=old_agent(20261101,'cuda');legacy.resume(a,args.checkpoint);cls=AuditedOld
    else:a=create_agent();resume(a,args.checkpoint);cls=AuditedSpeed
    args.output.mkdir(exist_ok=False)
    with StandaloneArena(args.output,headless=True) as arena:
        t=AtomicTransport(arena.port,hud=False)
        try:campaign(a,cls(t,detector,extractor),args.output/'cases',args.checkpoint,args.split,args.count,args.old_baseline,corners if args.corners else None)
        finally:t.close()

if __name__=='__main__':main()
