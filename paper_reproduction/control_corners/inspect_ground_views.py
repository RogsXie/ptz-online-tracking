"""Fixed diagnostic orientations for ground coverage, never learner samples."""
from pathlib import Path
import cv2
from integration.runtime import StandaloneArena,REPRO
from control_learning.environment import AtomicTransport
from control_corners.scenes import materialize,load_protocol
from control_corners.run import write_json


def main():
    out=REPRO/'verification/four-corners-v7-20261031/ground-view-limits'
    out.mkdir(exist_ok=False)
    scene=materialize('static_center','train',718950,out)
    protocol=load_protocol();poses=[]
    with StandaloneArena(out,headless=True) as arena:
        transport=AtomicTransport(arena.port,hud=False)
        try:
            transport.scenario(scene);transport.reset(718950)
            location=transport.diagnostics['camera_location']
            for pitch in [-80.,-60.,-40.]:
                for yaw in [-65.,0.,65.]:
                    transport.rpc(f'vset /ptz/view {pitch:.1f} {yaw:.1f} 60.0')
                    transport.capture()
                    assert transport.diagnostics['camera_location']==location
                    filename=f'pitch{pitch:.0f}-yaw{yaw:.0f}.jpg'
                    cv2.imwrite(str(out/filename),cv2.cvtColor(transport.rgb,cv2.COLOR_RGB2BGR))
                    poses.append({'pitch':pitch,'yaw':yaw,'image':filename,'camera_location':location,'rotation':transport.diagnostics['rotation']})
        finally:transport.close()
    write_json(out/'report.json',{'scope':'Nine fixed diagnostic views; no learning, target-directed action or control success claim',
        'training_updates':0,'poses':poses,'camera_contract':protocol['control_execution']})


if __name__=='__main__':main()
