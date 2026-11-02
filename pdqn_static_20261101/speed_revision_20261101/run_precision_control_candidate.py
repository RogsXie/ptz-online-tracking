"""Current validated PTZ control v3; original perception and old weights preserved."""
import speed_bootstrap
from speed_bootstrap import REV
import sys
from threadpoolctl import threadpool_limits


if __name__=='__main__':
    checkpoint=REV/'unified_motion_20261101/control-v3-turn-training-20261102/deploy.pt'
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    sys.argv=[sys.argv[0],'--scene','viaduct30','--seconds','28','--checkpoint',str(checkpoint),*sys.argv[1:]]
    from live_fast_online_results import main
    with threadpool_limits(limits=1,user_api='blas'):main()
