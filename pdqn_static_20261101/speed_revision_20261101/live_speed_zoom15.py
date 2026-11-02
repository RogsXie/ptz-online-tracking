"""User-requested 3x FOV zoom slew, same learned per-command zoom amount.

The existing frozen P-DQN live loop is reused. This changes the scene's actual
zoom rate and its constant raw-to-seconds conversion together, so a selected
raw zoom parameter still produces the same FOV increment. No video speed-up.
"""
import speed_bootstrap
import live_speed_v5 as runner
from scenes import materialize as base_materialize

ZOOM_RATE=15.
original_write=runner.write

def materialize(folder,item):
    rates=(4.,2.,5.) if item.get('execution_profile')=='released' else (25.,25.,ZOOM_RATE)
    return base_materialize(folder,item,rates)

def write(path,value):
    if path.name=='configuration.json' and value['execution_profile']!='released':
        value.update(execution_profile='pan25_tilt25_zoom15_calibrated_development_demo',
            rates=[25.,25.,ZOOM_RATE],zoom_rate_multiplier=3.,
            zoom_mapping_note='Actual FOV slew5to15deg/s; raw zoom duration.2to1/15s, same per-commandFOV increment. Frozen policy; no retraining or video acceleration.',
            qualification='Development video only; remaining validation canceled at user request')
    return original_write(path,value)

if __name__=='__main__':
    runner.materialize=materialize
    runner.SECONDS_PER_RAW=(.1,.016,.016,.032,.032,1./ZOOM_RATE,1./ZOOM_RATE)
    runner.write=write
    runner.main()
