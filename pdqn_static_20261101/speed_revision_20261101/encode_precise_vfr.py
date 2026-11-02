"""Preserve every real frame and its wall timestamps at millisecond precision.

ffconcat's default JPEG demuxer uses a25Hz time base, silently merging close
timestamps in VFR output. Set each JPEG's demuxer rate explicitly and pass
the actual capture timestamps through to the H264 encoder.
"""
import speed_bootstrap
from speed_bootstrap import REPRO
import json,subprocess,sys
from pathlib import Path

def encode(output,frames,name='live-detection-vfr.mp4'):
    sys.path.insert(0,str(REPRO/'verification/value-redesign-v10-20261031/video-tools'))
    from imageio_ffmpeg import get_ffmpeg_exe
    parts=['ffconcat version 1.0']
    for i,row in enumerate(frames):
        parts.extend([f"file 'frames/{i:05d}.jpg'",'option framerate 1000'])
        if i+1<len(frames):parts.append(f"duration {frames[i+1]['wall_s']-row['wall_s']:.9f}")
    manifest=output/'frames-precise.ffconcat';manifest.write_text('\n'.join(parts)+'\n',encoding='utf-8')
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    result=subprocess.run([get_ffmpeg_exe(),'-y','-loglevel','error','-safe','0','-f','concat','-i',str(manifest),
        '-fps_mode','passthrough','-enc_time_base','1:1000','-c:v','libx264','-crf','18','-preset','fast',
        '-threads','2','-pix_fmt','yuv420p','-video_track_timescale','1000000','-movflags','+faststart',str(output/name)],startupinfo=startup,capture_output=True)
    if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace'))
    import cv2
    v=cv2.VideoCapture(str(output/name));count=int(v.get(cv2.CAP_PROP_FRAME_COUNT));fps=v.get(cv2.CAP_PROP_FPS);v.release()
    assert count==len(frames),(count,len(frames))
    assert abs(count/fps-(frames[-1]['wall_s']-frames[0]['wall_s']))<.2
    return dict(frames=count,average_fps=fps,duration_s=count/fps,timebase='1ms input / encoder;1us MP4 track',no_dropped_or_synthesized_frames=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--timestamps',type=Path,required=True);p.add_argument('--name',default='live-detection-vfr.mp4');a=p.parse_args()
    fs=[json.loads(s) for s in a.timestamps.read_text().splitlines()];print(json.dumps(encode(a.folder.resolve(),fs,a.name)))
