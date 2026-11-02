"""Full action-end RGB recordings at one frame per .04 simulation seconds."""
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np

TOOLS=Path(__file__).resolve().parents[1]/'verification/value-redesign-v10-20261031/video-tools'
sys.path.insert(0,str(TOOLS))
from imageio_ffmpeg import get_ffmpeg_exe


class Stream:
    def __init__(self,path,size):
        self.log=path.with_suffix('.encode.log').open('wb')
        startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
        self.p=subprocess.Popen([get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
            '-s',f'{size[0]}x{size[1]}','-r','25','-i','pipe:0','-an','-c:v','libx264','-preset','fast',
            '-crf','18','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],
            stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.log,startupinfo=startup)

    def write(self,rgb):self.p.stdin.write(np.ascontiguousarray(rgb).tobytes())

    def close(self):
        self.p.stdin.close();code=self.p.wait(timeout=60);self.log.close()
        if code:raise RuntimeError('Video encoder failed')


class CaseVideo:
    def __init__(self,folder,seed):
        self.raw=Stream(folder/'sensor-rgb.mp4',(960,540))
        try:self.shown=Stream(folder/'policy-boxed.mp4',(960,604))
        except BaseException:
            self.raw.close()
            raise
        self.seed=seed;self.frames=0

    def write(self,rgb,row):
        self.raw.write(rgb)
        canvas=np.zeros((604,960,3),dtype=np.uint8);canvas[64:]=rgb
        bbox=row.get('selected_bbox')
        if bbox:
            x1,y1,x2,y2=(np.array(bbox)*[960,540,960,540]).astype(int)
            cv2.rectangle(canvas,(x1,y1+64),(x2,y2+64),(50,240,100),2)
        cv2.drawMarker(canvas,(480,334),(255,175,70),cv2.MARKER_CROSS,18,1)
        names=['Stay','Down','Up','Right','Left','Zoom in','Zoom out']
        title=f'v15 joint precision | v13 RGB response | seed {self.seed} | frozen planner | t={row["sim_time_s"]:.2f}s'
        center='lost' if row['center_error'] is None else f'{row["center_error"]:.4f}'
        area='lost' if row['area_ratio'] is None else f'{100*row["area_ratio"]:.2f}%'
        text=f'{names[row["action_type"]]} | center={center} | area={area} (target 4%) | FOV={row["fov"]:.2f} | 25fps sim time'
        cv2.putText(canvas,title,(12,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,245,250),1,cv2.LINE_AA)
        cv2.putText(canvas,text,(12,49),cv2.FONT_HERSHEY_SIMPLEX,.48,(200,220,235),1,cv2.LINE_AA)
        self.shown.write(canvas);self.frames+=1

    def close(self):
        try:self.raw.close()
        finally:self.shown.close()
