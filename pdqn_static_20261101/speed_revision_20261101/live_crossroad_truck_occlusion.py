"""Native box truck briefly occludes red; existing blue/purple and distance line remain."""
import speed_bootstrap
import json, copy, shutil
from pathlib import Path
import live_crossroad_purple_following as previous

runner = previous.runner

def materialize(folder, item):
    path = previous.materialize(folder, item)
    cfg = json.loads(path.read_text(encoding='utf-8'))
    truck = copy.deepcopy(cfg['targets'][0])
    truck.update(id='truck_foreground_occluder', vehicle_kind='box_truck',
                 color_rgb=[.85, .34, .025], scale=1.)
    truck['path'] = dict(type='line', origin_cm=[5900., 6500., 8.],
                         velocity_cm_s=[0., -1000., 0.],
                         speed_profile=[[0., 1.], [17., 0.]],
                         phase_s=0., random_phase=False)
    cfg['targets'].append(truck)
    runner.demo.MOTION.update(scenario='crossroad_truck_occlusion', truck_traffic=dict(
        description='Native 11.8m box truck, 2.8m wide and 4.47m tall. Oncoming adjacent lane, passes in front of red near7s. No policy occlusion schedule/identity input.',
        speed_m_s=10., path=truck['path']))
    cfg.update(name='user_crossroad_native_truck_occlusion', motion_schedule=runner.demo.MOTION,
        features=[*cfg['features'], 'native_box_truck', 'physical_foreground_occlusion'],
        scope_note='Adds native moving opaque truck geometry only. Existing frozen RGB perception/controller retained; no GT target selection, no synthetic cover or predicted boxes drawn through occlusion.')
    path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    shutil.copy2(Path(__file__), folder/'frozen-sources'/Path(__file__).name)
    return path

if __name__ == '__main__':
    runner.cross.materialize = materialize
    runner.write = previous.write
    runner.main()
