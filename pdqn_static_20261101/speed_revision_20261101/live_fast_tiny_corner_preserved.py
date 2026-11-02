"""Same tiny starting view, reachable limits, faster rates, original amplitudes."""
import speed_bootstrap
import live_fast_tiny_corner as fast
import live_speed_v5 as runner
import live_tiny_corner_zoom15 as tiny

def write(path,value):
    if path.name=='configuration.json':
        value.update(execution_profile='fast_tiny_pan180_tilt100_zoom100_original_amplitudes',rates=list(fast.RATES),
            execution_calibration='Original learned raw command amplitudes preserved exactly; only actualexecutionrate and reciprocal duration mapping changed. No durationstretch.',
            qualification='Fresh development demonstration only; no resumedvalidation or networkupdates',
            scope='Same0.22percent upperleft initialview, pitchlimit-85..-30; exclusive7actions, no alignmentrule or simultaneousmotion')
    return fast.original_write(path,value)

if __name__=='__main__':
    runner.layout=tiny.layout
    runner.materialize=fast.materialize
    runner.resume=fast.original_resume
    runner.SECONDS_PER_RAW=(.1,.4/100.,.4/100.,.8/180.,.8/180.,1./100.,1./100.)
    runner.write=write
    runner.main()
