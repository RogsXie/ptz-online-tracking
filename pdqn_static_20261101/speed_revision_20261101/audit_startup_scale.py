"""Compare first-lock zoom timing from actual live logs; no rendering/inference."""
import json
from pathlib import Path
from audit_continuous_online import summarize,read

ROOT=Path(__file__).resolve().parent

def timing(folder):
    decisions=read(folder/'decisions.jsonl')
    observations=read(folder/'perception-online.jsonl')
    zoom=[r for r in decisions if r['action']==5]
    startup=[r for r in decisions if r['area_constraint']['reason']=='locked_startup_zoom_in']
    reached=[r for r in observations if r['visible'] and r['area']>=.0144]
    first=[r for r in observations if r['sim_time']<6 and r['visible']]
    return dict(first_zoom_native_start_s=zoom[0]['command_execution']['native_sim_s'] if zoom else None,
        first_reached_1p44_percent_area_s=reached[0]['sim_time'] if reached else None,
        startup_zoom_pulses=len(startup),maximum_startup_zoom_duration_s=max([r['duration'] for r in startup],default=0.),
        startup_zoom_steps=[dict(native_start_s=r['command_execution']['native_sim_s'],duration_s=r['duration'],
            input_area=r['area_constraint']['input_area'],input_fov_deg=r['area_constraint']['input_fov_deg'],
            input_error_px=r['area_constraint']['predicted_error_px']) for r in startup],
        max_visible_area_first6s=max([r['area'] for r in first],default=0.))

if __name__=='__main__':
    base=ROOT/'unified_motion_20261101'
    old=base/'user-fast-genuine-online-v2-video';new=base/'user-startup-active-zoom-v1-video'
    report=dict(previous=timing(old),updated=timing(new),quality_and_online_proof=summarize(new),
        change='Active per-target first-lock scale acquisition toward1.6%;40px rough-centre gate;6deg max exclusive zoom step; stop at>=1.44%. Existing2% upper guard, detector/ReID and frozen policy weights unchanged.',
        scope='One 30s live scene. No offline detection, reassociation or box redrawing. GT read only after the run for scoring.')
    (new/'startup-scale-result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
