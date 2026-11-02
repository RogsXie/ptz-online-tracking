"""Explicit zoom-dependent actuator precision for the same tiny-corner demo.

P-DQN directions and raw durations are unmodified. Only angular actuator gain
is scheduled by camera optics: tan(FOV/2)^2 relative to the initial66deg view.
No target error, target coordinates, GT, direction rule, or parallel actions
are used by the execution adapter. This is an engineering extension.
"""
import speed_bootstrap
import json,math,time
from pathlib import Path
import live_speed_v5 as runner
import live_fast_tiny_corner as fast
import live_tiny_corner_zoom15 as tiny

REFERENCE_FOV=66.

class OpticsPrecisionMixin:
    def rpc(self,command):
        if command.startswith('vset /ptz/play '):
            pan,tilt,zoom,duration=map(float,command.split()[2:])
            fov=float(self.diagnostics['fov'])
            gain=min(1.,2.*(math.tan(math.radians(fov/2))/math.tan(math.radians(REFERENCE_FOV/2)))**2)
            zoom_gain=min(1.,fov/REFERENCE_FOV)
            executed=f'vset /ptz/play {pan*gain:.12f} {tilt*gain:.12f} {zoom*zoom_gain:.12f} {duration:.12f}'
            if not hasattr(self,'precision_log'):
                import sys
                folder=Path(sys.argv[sys.argv.index('--output')+1])
                self.precision_log=(folder/'actuator-precision.jsonl').open('w',encoding='utf-8',buffering=1)
            self.precision_log.write(json.dumps(dict(perf_counter_s=time.perf_counter(),before_sim_time=self.diagnostics['sim_time'],
                fov=fov,angular_gain=gain,learned_direction_command=command,executed_command=executed,
                actual_pan_deg_s=pan*gain*180.,actual_tilt_deg_s=tilt*gain*100.,actual_zoom_fov_deg_s=zoom*zoom_gain*100.,duration=duration))+'\n')
            command=executed
        return super().rpc(command)

    def close(self):
        try:return super().close()
        finally:
            if hasattr(self,'precision_log'):self.precision_log.close()

class OpticsTransport(OpticsPrecisionMixin,runner.AtomicTransport):pass
class OpticsDisplayTransport(OpticsPrecisionMixin,runner.DisplayTransport):pass

def write(path,value):
    if path.name=='configuration.json':
        value.update(execution_profile='tiny_corner_optics_scheduled_angular_and_zoom_precision',rates=list(fast.RATES),
            execution_calibration='Source directions/raw durations unchanged; angularrate multiplier=min(1,2*(tan(FOV/2)/tan(66deg/2))**2). Zoomrate multiplier=min(1,FOV/66), keeping zoomincrements proportionalto currentFOV. Actualcommands logged inactuator-precision.jsonl.',
            actuator_inputs='CameraFOV used only by disclosed actuator gain schedule; no bboxerror,GTorcoordinates and no extra networkinputs.',
            qualification='Fresh development demonstration; no training or restartedvalidation',
            scope='Same0.22percent corner, reachablepitch-85..-30, exclusive7actions. Explicit engineered optics compensation, not exactoriginalpaperprotocol.')
    return fast.original_write(path,value)

if __name__=='__main__':
    runner.layout=tiny.layout
    runner.materialize=fast.materialize
    runner.resume=fast.original_resume
    runner.AtomicTransport=OpticsTransport
    runner.DisplayTransport=OpticsDisplayTransport
    runner.SECONDS_PER_RAW=(.1,.4/100.,.4/100.,.8/180.,.8/180.,1./100.,1./100.)
    runner.write=write
    runner.main()
