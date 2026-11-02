"""Short RGB LK bridge, checked by forward/back flow and warped appearance."""
import cv2,numpy as np
from unified_perception import bearing,project
from hascam.detection import Detection

class VisualBridge:
    def __init__(self,max_seconds=.35):
        self.max_seconds=max_seconds;self.gray=None;self.reason='unseeded';self.proposal=None

    def seed(self,rgb,box,diag,label='car',class_id=0):
        if rgb is None:return
        h,w=rgb.shape[:2];b=np.array(box)*[w,h,w,h]
        mask=np.zeros((h,w),np.uint8);x1,y1,x2,y2=b
        mx,my=.1*(x2-x1),.1*(y2-y1)
        cv2.rectangle(mask,(max(0,int(x1+mx)),max(0,int(y1+my))),(min(w-1,int(x2-mx)),min(h-1,int(y2-my))),255,-1)
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        points=cv2.goodFeaturesToTrack(gray,maxCorners=40,qualityLevel=.015,minDistance=3,mask=mask,blockSize=3)
        if points is None or len(points)<5:self.gray=None;self.reason='too_few_points';return
        self.gray=gray;self.points=points;self.box=np.array(box);self.diag={k:diag[k] for k in ['sim_time','fov','rotation']}
        self.seed_time=diag['sim_time'];self.label=label;self.class_id=class_id

    def propose(self,rgb,diag):
        self.proposal=None
        if rgb is None or self.gray is None:return None
        age=diag['sim_time']-self.seed_time
        if age<0 or age>self.max_seconds:self.reason='bridge_expired';return None
        h,w=rgb.shape[:2];gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        guesses=np.array([np.asarray(project(bearing((p[0]/w,p[1]/h),self.diag),diag))*[w,h] for p in self.points[:,0]],np.float32).reshape(-1,1,2)
        current,ok,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,self.points,guesses,winSize=(21,21),maxLevel=3,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        if current is None:self.reason='flow_failed';return None
        back,back_ok,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,current,self.points.copy(),winSize=(21,21),maxLevel=3,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        if back is None:self.reason='backward_failed';return None
        fb=np.linalg.norm(back[:,0]-self.points[:,0],axis=1)
        good=(ok[:,0]>0)&(back_ok[:,0]>0)&(fb<1.)
        if good.sum()<5 or good.mean()<.65:self.reason='flow_inconsistent';return None
        matrix,inliers=cv2.estimateAffinePartial2D(self.points[good,0],current[good,0],method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=500)
        if matrix is None or inliers.mean()<.65:self.reason='affine_inconsistent';return None
        scale=float(np.linalg.norm(matrix[:,0]));expected=np.tan(np.radians(self.diag['fov']/2))/np.tan(np.radians(diag['fov']/2))
        if not .75<=scale/expected<=1.33:self.reason='scale_inconsistent';return None
        x1,y1,x2,y2=self.box*[w,h,w,h]
        corners=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],np.float32)
        moved=cv2.transform(corners[None],matrix)[0]
        b=np.r_[moved.min(0),moved.max(0)]/[w,h,w,h];b=np.clip(b,0,1)
        if b[2]<=b[0] or b[3]<=b[1]:self.reason='invalid_box';return None
        warped=cv2.warpAffine(gray,matrix,(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
        left,top=max(0,int(x1)),max(0,int(y1));right,bottom=min(w,int(x2)),min(h,int(y2))
        a=self.gray[top:bottom,left:right];z=warped[top:bottom,left:right]
        if a.size<64 or np.std(a)<4:self.reason='weak_texture';return None
        ncc=float(cv2.matchTemplate(z,a,cv2.TM_CCOEFF_NORMED)[0,0])
        quality=min(ncc,float(good.mean()),float(inliers.mean()))
        if ncc<.72 or quality<.65:self.reason='appearance_changed';return None
        self.proposal=dict(gray=gray,points=current[good],box=b,diag={k:diag[k] for k in ['sim_time','fov','rotation']},quality=quality,ncc=ncc,fb_error=float(np.median(fb[good])))
        self.reason='verified'
        return Detection(tuple(b),min(.99,max(0.,quality)),self.class_id,self.label)

    def accept(self):
        if self.proposal is None:return
        p=self.proposal;self.gray=p['gray'];self.points=p['points'];self.box=p['box'];self.diag=p['diag']
