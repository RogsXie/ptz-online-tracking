"""RGB-error axis scheduling; no scene schedule or target-state inputs."""
import numpy as np

ACQUIRE_CENTER_PX=12.
ZOOM_FEEDBACK_TOLERANCE_PX=20.


def preferred_axis(error,valid,last_axis=None,last_times=None,now=None):
    magnitude=np.abs(np.asarray(error,float))
    axes=[i for i,actions in enumerate(([3,4],[1,2])) if any(valid[a] for a in actions)]
    if not axes:return None
    major=max(axes,key=lambda i:magnitude[i])
    if magnitude[major]<=8.:return None
    if last_times is not None and now is not None:
        waiting=[i for i in axes if magnitude[i]>=8. and now-last_times.get(i,now)>=.28]
        if waiting:return max(waiting,key=lambda i:now-last_times.get(i,now))
    if len(axes)==2 and last_axis in axes:
        other=1-last_axis
        # Comparable errors share feedback cycles. A clearly larger error
        # always wins, regardless of the network's fixed action preference.
        if magnitude[other]>=8. and magnitude[other]>=magnitude[major]/1.15:return other
    return major


def pulse_error(error,pixel_velocity,horizon):
    magnitude=np.abs(error)
    extra=np.clip(np.sign(error)*np.asarray(pixel_velocity)*min(.08,.5*horizon),0.,np.maximum(4.,.5*magnitude))
    return np.sign(error)*(magnitude+extra)
