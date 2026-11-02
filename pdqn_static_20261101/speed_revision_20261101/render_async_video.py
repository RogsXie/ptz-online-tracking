"""Same RGB detector overlay, with lossless frame-count/timestamp export."""
import speed_bootstrap
import render_unified_video as renderer
from encode_precise_vfr import encode
if __name__=='__main__':
    renderer.encode=encode
    renderer.main()
