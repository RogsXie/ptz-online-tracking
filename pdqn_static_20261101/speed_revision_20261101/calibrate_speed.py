"""Real UE pulse calibration; these are hardware probes, never policy data."""
import speed_bootstrap
from speed_bootstrap import REV,REPRO
import json,time,warnings
import cv2,torch
from scenes import layout,materialize
from calibrate import measure,write
from train import exclusive
from control_model_v13.detector import create_detector
from integration.experiment import make_extractor
from integration.runtime import StandaloneArena
from author_aligned.environment import AtomicTransport
from integration.transport import ACTION_RATES

def main():
    exclusive();warnings.filterwarnings('ignore');torch.set_num_threads(4)
    out=REV/'calibration-v1';out.mkdir(exist_ok=False)
    detector=create_detector(checkpoint=REPRO/'weights/yolo26s_ue_overhead_v1.pt',device='cuda',threshold=.15,imgsz=960)
    assert detector.provenance['sha256']=='1dd0de55c1791dce289cc7dc1d8eb6c50e488b26d38941f4c20845a6cadb5ae2'
    detector.warmup();extractor=make_extractor('cuda');assert extractor.sha256=='57fb9c17d88911ea64390bf5427f43511435e7f88f6eed9dbc969d4b611e53cd'
    results=[]
    with StandaloneArena(out,headless=True) as arena:
        t=AtomicTransport(arena.port,hud=False)
        try:
            for profile,rates in [('released',(4.,2.,5.)),('paper25',(25.,25.,5.))]:
                for fov in (14.,20.,30.,60.):
                    item=layout('calibration',0)|dict(planned_center=[.5,.5],fov_deg=fov)
                    scene=materialize(out/f'{profile}-fov{fov:g}',item,rates)
                    for action in (1,2,3,4,5,6):
                        for duration in (.02,.04,.06,.12):
                            t.scenario(scene);t.reset(item['seed']);before=measure(t,detector,extractor);diag0=t.diagnostics.copy()
                            pan,tilt,zoom=ACTION_RATES[action];start=time.perf_counter();t.rpc(f'vset /ptz/step {pan} {tilt} {zoom} {duration:.12f}');t.capture()
                            after=measure(t,detector,extractor)
                            result=dict(profile=profile,rates=rates,fov=fov,action=action,requested_seconds=duration,
                                before=before,after=after,rotation_before=diag0['rotation'],rotation_after=t.diagnostics['rotation'],
                                fov_after=t.diagnostics['fov'],actual_seconds=t.diagnostics['sim_time']-diag0['sim_time'],
                                wall_s=time.perf_counter()-start)
                            assert abs(result['actual_seconds']-duration)<1e-5
                            if before['center'] and after['center']:
                                result['center_delta_pixels']=[(after['center'][0]-before['center'][0])*960,(after['center'][1]-before['center'][1])*540]
                            results.append(result);write(out/'responses.json',results)
                    print('CALIBRATED',profile,'FOV',fov,'probes',len(results),flush=True)
            # Native realtime floor is tested independently of atomic stepping.
            item=layout('calibration',0)|dict(planned_center=[.5,.5],fov_deg=30.)
            scene=materialize(out/'realtime-floor',item,(25.,25.,5.));floors=[]
            for requested in (.002,.004,.01,.02,.04,.06):
                t.scenario(scene);t.reset(item['seed']);t.rpc('vset /ptz/realtime 1')
                reply=t.rpc(f'vset /ptz/play 1 0 0 {requested:.12f}')
                while True:
                    t.capture()
                    if t.diagnostics['action_remaining_seconds']<=1e-6:break
                floors.append(dict(requested=requested,returned_remaining=reply['action_remaining_seconds'],rotation=t.diagnostics['rotation']))
                t.rpc('vset /ptz/realtime 0')
            write(out/'realtime-floor.json',floors)
        finally:t.close()
    summary=dict(real_probes=len(results),training_transitions=0,realtime_floor=floors,
        paper25_min_source_pulse=[r for r in results if r['profile']=='paper25' and r['requested_seconds']==.06 and r['action']==2],
        perception_frozen=True,scope='Independent calibration commands; no trained policy, no teacher transitions')
    write(out/'summary.json',summary)

if __name__=='__main__':main()
