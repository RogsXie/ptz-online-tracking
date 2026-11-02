"""Same isolated UE process, local writable DDC fallback for this launch only."""
import speed_bootstrap
import subprocess,sys
from integration import runtime

class StreamArena(runtime.StandaloneArena):
    def __enter__(self):
        self.output.mkdir(parents=True,exist_ok=True)
        project=runtime.UE_PROJECT
        map_name='/Game/PTZ/Maps/PTZResearch'
        command=[str(runtime.UE_EDITOR),str(project),map_name,'-game','-windowed','-ResX=960','-ResY=540','-NoSplash',
            f'-cvport={self.port}',f'-abslog={self.output / "ue.log"}','-RenderOffscreen','-unattended','-ddc=InstalledNoZenLocalFallback',
            '-NoVSync','-ExecCmds=r.VSync 0,r.OneFrameThreadLag 1,t.MaxFPS 0']
        startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
        self.process=subprocess.Popen(command,cwd=str(project.parent),startupinfo=startup,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:runtime.wait_for_arena(self.process,self.port)
        except BaseException:self.__exit__(*sys.exc_info());raise
        return self
