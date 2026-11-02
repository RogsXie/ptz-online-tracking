"""Run the independently qualified speed release in the existing CUDA environment."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
from datetime import datetime
ROOT=Path(__file__).resolve().parent
PY=Path('C:/Users/12879/.conda/envs/50mamba/python.exe')
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    if Path(sys.executable).resolve()!=PY.resolve():
        if not PY.is_file():raise RuntimeError(f'Existing interpreter required: {PY}')
        raise SystemExit(subprocess.run([str(PY),'-B',str(Path(__file__).resolve()),*sys.argv[1:]]).returncode)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--headless',action='store_true');p.add_argument('--case',type=int,default=7);p.add_argument('--seconds',type=float,default=50.);args=p.parse_args()
    if not 0<=args.case<80:raise ValueError('Known development case index must be0..79')
    marker=ROOT/'qualified-speed-release.json'
    if not marker.exists():raise SystemExit('速度修订尚未完成独立验收。现有合格入口仍为上级目录的 run_in_pycharm.py。')
    release=load(marker);cp=Path(release['checkpoint']);assert hashlib.sha256(cp.read_bytes()).hexdigest()==release['checkpoint_sha256']
    for path,h in load(release['runtime_manifest'])['files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h:raise RuntimeError(f'Qualified runtime changed: {path}')
    for key in ('validation_report','final_test_report'):
        s=load(release[key]);assert s['qualification']['qualified'] and s['qualification']['count']==160 and s['configuration']['checkpoint_sha256']==release['checkpoint_sha256']
    output=ROOT/'demo-runs'/datetime.now().strftime('%Y%m%d-%H%M%S-%f');output.parent.mkdir(exist_ok=True)
    cmd=[str(PY),'-B',str(ROOT/release['live_entry']),'--checkpoint',str(cp),'--output',str(output),'--index',str(args.case),'--seconds',str(args.seconds)]
    if not args.headless:cmd.append('--visible')
    print('运行新的真实 PTZ 闭环：',output,flush=True)
    result=subprocess.run(cmd)
    if result.returncode:raise SystemExit(result.returncode)
    result=subprocess.run([str(PY),'-B',str(ROOT.parent/'render_detection_video.py'),str(output)])
    if result.returncode:raise SystemExit(result.returncode)
    print('带框视频：',output/'detection-overlay/live-detection-vfr.mp4',flush=True)
if __name__=='__main__':main()
