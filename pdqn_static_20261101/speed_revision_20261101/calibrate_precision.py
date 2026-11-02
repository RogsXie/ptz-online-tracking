"""Verify short-pulse atomic and realtime behavior in the isolated native copy."""
import speed_bootstrap
from speed_bootstrap import REV,REPRO,configure_precision_project
import time,warnings,torch
from scenes import layout,materialize
from calibrate import measure,write
from train import exclusive
from control_model_v13.detector import create_detector
from integration.experiment import make_extractor
from integration.runtime import StandaloneArena
from author_aligned.environment import AtomicTransport
from integration.transport import ACTION_RATES

def main():
    exclusive();configure_precision_project();warnings.filterwarnings('ignore');torch.set_num_threads(4)
    out=REV/'precision-calibration-v2';out.mkdir(exist_ok=False)
    detector=create_detector(checkpoint=REPRO/'weights/yolo26s_ue_overhead_v1.pt',device='cuda',threshold=.15,imgsz=960);detector.warmup()
    extractor=make_extractor('cuda');rows=[];floors=[]
    with StandaloneArena(out,headless=True) as arena:
        t=AtomicTransport(arena.port,hud=False)
        try:
            for fov in (14.,20.,30.,60.):
                item=layout('calibration',0)|dict(planned_center=[.5,.5],fov_deg=fov)
                scene=materialize(out/f'fov{fov:g}',item,(25.,25.,5.))
                for action in (1,2,3,4):
                    for dt in (.002,.004,.01,.06):
                        t.scenario(scene);t.reset(item['seed']);before=measure(t,detector,extractor);d0=t.diagnostics.copy()
                        pan,tilt,zoom=ACTION_RATES[action];t.rpc(f'vset /ptz/step {pan} {tilt} {zoom} {dt:.12f}');t.capture();after=measure(t,detector,extractor)
                        assert abs(t.diagnostics['sim_time']-d0['sim_time']-dt)<1e-6
                        rows.append(dict(fov=fov,action=action,duration=dt,before=before,after=after,
                            rotation_before=d0['rotation'],rotation_after=t.diagnostics['rotation'],
                            center_delta_pixels=[(after['center'][0]-before['center'][0])*960,(after['center'][1]-before['center'][1])*540] if before['center'] and after['center'] else None))
                print('PRECISION CALIBRATED',fov,len(rows),flush=True)
            for dt in (.002,.004,.01,.04,.06):
                t.scenario(scene);t.reset(item['seed']);t.rpc('vset /ptz/realtime 1');reply=t.rpc(f'vset /ptz/play 1 0 0 {dt:.12f}')
                while True:
                    t.capture()
                    if t.diagnostics['action_remaining_seconds']<=1e-6:break
                floors.append(dict(requested=dt,returned_remaining=reply['action_remaining_seconds'],rotation=t.diagnostics['rotation']))
                assert abs(t.diagnostics['rotation'][1]-25*dt)<1e-5
                t.rpc('vset /ptz/realtime 0')
        finally:t.close()
    write(out/'responses.json',rows);write(out/'realtime-floor.json',floors)
    write(out/'summary.json',dict(passed=True,real_atomic_probes=len(rows),real_realtime_probes=len(floors),training_transitions=0,
        fine_tilt=[r for r in rows if r['action']==2 and r['duration']==.002],native_short_pulse_verified=True))

if __name__=='__main__':main()
