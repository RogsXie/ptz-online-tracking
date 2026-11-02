"""All real frames, actual source timestamps, nominal60Hz codec timing."""
import speed_bootstrap
from speed_bootstrap import REPRO
import sys,subprocess
import cv2

def encode(output,frames,name='live-detection-vfr.mp4',duration_tolerance_s=.2,bframes=None):
    sys.path.insert(0,str(REPRO/'verification/value-redesign-v10-20261031/video-tools'))
    from imageio_ffmpeg import get_ffmpeg_exe
    parts=['ffconcat version 1.0']
    for i,row in enumerate(frames):
        parts.extend([f"file 'frames/{i:05d}.jpg'",'option framerate 1000'])
        if i+1<len(frames):parts.append(f"duration {frames[i+1]['wall_s']-row['wall_s']:.9f}")
    manifest=output/'frames-source60.ffconcat';manifest.write_text('\n'.join(parts)+'\n',encoding='utf-8')
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    extra=['-bf',str(bframes)] if bframes is not None else []
    r=subprocess.run([get_ffmpeg_exe(),'-y','-loglevel','error','-safe','0','-f','concat','-i',str(manifest),
        '-fps_mode','passthrough','-enc_time_base','1:1000','-c:v','libx264','-x264-params','fps=60/1','-level:v','4.1','-crf','18','-preset','fast','-threads','2',
        '-pix_fmt','yuv420p','-video_track_timescale','1000000',*extra,'-movflags','+faststart',str(output/name)],startupinfo=startup,capture_output=True)
    if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
    v=cv2.VideoCapture(str(output/name));n=int(v.get(cv2.CAP_PROP_FRAME_COUNT));fps=v.get(cv2.CAP_PROP_FPS);v.release()
    assert n==len(frames),(n,len(frames))
    assert abs(n/fps-(frames[-1]['wall_s']-frames[0]['wall_s']))<duration_tolerance_s
    return dict(encoded_frames=n,encoded_average_fps=fps,encoded_duration_s=n/fps,no_duplicate_interpolated_or_dropped_frames=True)

if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--timestamps',type=Path,required=True);p.add_argument('--name',default='live-detection-vfr.mp4');a=p.parse_args()
    f=[json.loads(s) for s in a.timestamps.read_text().splitlines()];print(json.dumps(encode(a.folder.resolve(),f,a.name)))
