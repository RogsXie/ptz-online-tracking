"""Same-frame frozen RGB boxes with every real60Hz source image preserved."""
import speed_bootstrap
import render_unified_video as renderer
from encode_60fps_vfr import encode
if __name__=='__main__':
    renderer.encode=encode
    renderer.main()
