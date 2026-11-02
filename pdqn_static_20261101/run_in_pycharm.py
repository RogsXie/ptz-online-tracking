"""PyCharm entry for the independently qualified policy, once released.

Right-click Run using the existing 50mamba interpreter. This opens real UE PTZ
control and saves a new live video. It never trains or substitutes hand rules.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
from datetime import datetime

HERE=Path(__file__).resolve().parent
PYTHON=Path(r'C:/Users/12879/.conda/envs/50mamba/python.exe')

def main():
    if Path(sys.executable).resolve()!=PYTHON.resolve():
        if not PYTHON.is_file():raise RuntimeError(f'Existing CUDA interpreter is required: {PYTHON}')
        result=subprocess.run([str(PYTHON),'-B',str(Path(__file__).resolve()),*sys.argv[1:]])
        raise SystemExit(result.returncode)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless',action='store_true',help='Render and record without opening the UE window')
    parser.add_argument('--case',type=int,help='Optional validation case index0..159')
    parser.add_argument('--seconds',type=float,default=50.)
    args=parser.parse_args()
    release_path=HERE/'qualified-release.json'
    if not release_path.exists():
        raise SystemExit('策略尚未通过完整独立验收，暂未发布默认演示模型。训练正在持续推进；已有开发视频保存在early-live-case035。')
    release=json.loads(release_path.read_text(encoding='utf-8'))
    checkpoint=Path(release['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=release['checkpoint_sha256']:
        raise RuntimeError('Qualified checkpoint content changed')
    manifest=json.loads(Path(release['runtime_manifest']).read_text(encoding='utf-8'))
    for path,expected in manifest['files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Qualified runtime or frozen perception weights changed: {path}')
    for key in ('validation_report','final_test_report'):
        report=json.loads(Path(release[key]).read_text(encoding='utf-8'))
        if not report['qualification']['qualified'] or report['qualification']['count']!=160:
            raise RuntimeError('Complete independent160-case qualification is required')
        if report['configuration']['checkpoint_sha256']!=release['checkpoint_sha256']:
            raise RuntimeError('Qualification and checkpoint differ')
    case=args.case if args.case is not None else release['default_demo_case']
    if not 0<=case<160:raise ValueError('Case index must be0..159')
    output=HERE/'demo-runs'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output.parent.mkdir(exist_ok=True)
    demo_entry=HERE/('live_demo.py' if args.headless else 'live_demo_visible.py')
    command=[str(PYTHON),'-B',str(demo_entry),'--checkpoint',str(checkpoint),
        '--output',str(output),'--index',str(case),'--split','validation','--seconds',str(args.seconds)]
    if not args.headless:command.append('--visible')
    print('正在运行真实PTZ闭环；输出目录：',output,flush=True)
    result=subprocess.run(command)
    if result.returncode:raise SystemExit(result.returncode)
    print('原始视频：',output/'live-sensor-vfr.mp4',flush=True)
    print('正在用冻结RGB检测器生成带框视频……',flush=True)
    overlay=subprocess.run([str(PYTHON),'-B',str(HERE/'render_detection_video.py'),str(output)])
    if overlay.returncode:raise SystemExit(overlay.returncode)
    print('带检测框视频：',output/'detection-overlay/live-detection-vfr.mp4',flush=True)

if __name__=='__main__':main()
