"""Image-only HUD refresh for the existing, unmodified live controller.

Native /ptz/play clears the last HUD detection. Restore that same last RGB
observation for display. Every motion command, returned state and policy call
is delegated unchanged to the qualified live_demo module.
"""
import bootstrap
import hashlib,json,sys
from pathlib import Path
import live_demo as runner

def display_command(row):
    box=row['bbox'] or [0.,0.,0.,0.]
    return 'vset /ptz/detection '+' '.join(f'{v:.12f}' for v in box)+f" {int(row['visible'])}"

class DisplayTransport(runner.AtomicTransport):
    def __init__(self,*args,**kwargs):
        self.last_display_command=None
        super().__init__(*args,**kwargs)

    def rpc(self,command):
        reply=super().rpc(command)
        if command.startswith('vset /ptz/detection '):self.last_display_command=command
        elif command.startswith(('vset /ptz/reset ','vset /ptz/scenario ')):self.last_display_command=None
        elif command.startswith('vset /ptz/play ') and self.last_display_command is not None:
            super().rpc(self.last_display_command)
        return reply

class DisplayEnvironment(runner.Environment):
    def reset(self,*args,**kwargs):
        state,row=super().reset(*args,**kwargs)
        self.t.rpc(display_command(row))
        return state,row

if __name__=='__main__':
    if '--visible' not in sys.argv:raise ValueError('This entry is exclusively the visible image-detection HUD adapter')
    runner.AtomicTransport=DisplayTransport;runner.Environment=DisplayEnvironment
    runner.main()
    folder=Path(sys.argv[sys.argv.index('--output')+1]);path=folder/'configuration.json'
    cfg=json.loads(path.read_text(encoding='utf-8'))
    cfg.update(visible_hud_adapter='Restore last RGB detection after nativeplay clears HUD; motion commands and policy states unchanged',
        visible_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        qualified_live_demo_sha256=hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest())
    path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
