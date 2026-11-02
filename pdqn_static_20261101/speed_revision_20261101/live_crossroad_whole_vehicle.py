"""Same truck occlusion scene with learned whole-truck RGB boxes."""
import speed_bootstrap
import json, shutil
from pathlib import Path
import live_crossroad_reacquire_area2 as previous
from whole_vehicle_frontend import frontend

runner = previous.runner

class Observation(previous.Observation):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.cfff.match_same_class = True

    def observe(self,*args,**kwargs):
        state,row = super().observe(*args,**kwargs)
        row.update(target_label=self.cfff.info.get('selected_label'),
                   target_class_id=self.cfff.info.get('selected_class_id'),
                   whole_vehicle_box=True)
        return state,row

def materialize(folder,item):
    path = previous.materialize(folder,item)
    # Continue the existing road past the truck's endpoint; actors keep identical paths.
    cfg = json.loads(path.read_text(encoding='utf-8'))
    cfg['occluders'].extend([
        dict(kind='road_surface',location_cm=[6000.,-11400.,.15],size_cm=[1200.,2800.,.2],color_rgb=[.13,.14,.14]),
        *[dict(kind='road_edge',location_cm=[6000.+s*550.,-11400.,.5],size_cm=[12.,2800.,.2],color_rgb=[.84,.84,.78]) for s in [-1,1]],
        *[dict(kind='road_centerline',location_cm=[6000.+s*8.,-11400.,.5],size_cm=[7.,2800.,.2],color_rgb=[.92,.68,.06]) for s in [-1,1]]])
    cfg['scope_note'] += ' Wholevehicle detector enabled; identical native vehicle paths. Road extended at truck endpoint.'
    path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    for n in ['live_crossroad_whole_vehicle.py','whole_vehicle_frontend.py','train_whole_vehicle_detector.py','confidence_reacquire.py']:
        shutil.copy2(Path(__file__).parent/n,folder/'frozen-sources'/n)
    return path

def write(path,value):
    if path.name=='configuration.json':
        value.update(whole_vehicle_detector=True,association_same_class_until_loss=True,
            truck_box_scope='Detector directly predicts whole truck, including cab and cargo. All selected-box centre/area/ReID crops use the same full RGB box. No GT size replacement.')
    return previous.write(path,value)

if __name__=='__main__':
    runner.core.frontend = frontend
    runner.cross.materialize = materialize
    runner.Observation = Observation
    runner.write = write
    runner.main()
