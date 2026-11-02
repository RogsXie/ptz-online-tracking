"""Serialize post-training screening and focused real-online comparisons."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();training=args.training.resolve();out=args.output.resolve();out.mkdir(exist_ok=False)
    def write(name,value):(out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    def run(label,script,*arguments):
        print('SERIAL CHECK START',label,flush=True);write('progress.json',dict(phase=label,started=time.time()))
        with (out/f'{label}.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(HERE/script),*map(str,arguments)],cwd=HERE,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('SERIAL CHECK COMPLETE',label,flush=True)
    try:
        while True:
            status=json.loads((training/'status.json').read_text()) if (training/'status.json').exists() else {}
            if status.get('status') in ['complete','interrupted_or_failed']:break
            time.sleep(10)
        if status['status']!='complete':raise RuntimeError('Training did not complete: '+status['status'])
        run('checkpoint-proof','check_fresh_checkpoint.py','--training',training,'--output',out/'checkpoint-proof.json')
        old=HERE/'unified_motion_20261101/pdqn_training_v1/last.pt'
        write('perception-scope.json',dict(detector='Original FP32',reid='Original fresh FP32',
            tracker='Original robust association',performance_experiments=False,
            reason='User requested camera-control reinforcement with perception fixed.'))
        runs=[('old-fixed-perception-control',old,[]),
            ('fresh-fixed-perception-control',training/'last.pt',[]),('random-fixed-perception-control',training/'initial-random.pt',[])]
        for label,checkpoint,flags in runs:
            run(label,'live_fast_online_results.py','--scene','viaduct30','--seconds',28,'--fps',30,'--perception-fps',30,
                '--checkpoint',checkpoint,'--output',out/label,*flags)
        heldouts=[]
        for label,checkpoint in [('old-heldout',old),('fresh-heldout',training/'last.pt')]:
            run(label,'evaluate_precision_policy.py','--checkpoint',checkpoint,'--output',out/label,'--steps',64)
            heldouts.append(out/label)
        run('summary','summarize_precision_attempt.py','--training',training,'--online',*[out/x[0] for x in runs],
            '--heldout',*heldouts,'--output',out/'comparison.json')
        run('camera-only','evaluate_camera_control_only.py','--runs',*[out/x[0] for x in runs],
            '--output',out/'camera-only-comparison.json')
        write('status.json',dict(status='complete',training=str(training),comparison=str(out/'comparison.json'),
            stable_default_checkpoint_replaced=False,private_repository_modified=False))
    except BaseException:
        write('status.json',dict(status='failed',traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
