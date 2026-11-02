"""Own and clean up one standalone UE process, leaving existing editors alone."""
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "paper_reproduction"
sys.path.insert(0, str(ROOT / "scripts"))
from run_in_pycharm import UE_EDITOR, UE_PROJECT, free_port, wait_for_arena


def new_run(name):
    path = REPRO / "runs" / name / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path.mkdir(parents=True)
    return path


class StandaloneArena(AbstractContextManager):
    def __init__(self, output, headless=False):
        self.output, self.headless = Path(output), headless
        self.port = free_port()
        self.process = None

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=True)
        project_map = UE_PROJECT.parent / "Content" / "PTZ" / "Maps" / "PTZResearch.umap"
        map_name = "/Game/PTZ/Maps/PTZResearch" if project_map.exists() else "/Engine/Maps/Templates/Template_Default"
        command = [str(UE_EDITOR), str(UE_PROJECT), map_name, "-game", "-windowed",
                   "-ResX=960", "-ResY=540", "-NoSplash", f"-cvport={self.port}",
                   f"-abslog={self.output / 'ue.log'}"]
        if self.headless:
            command += ["-RenderOffscreen", "-unattended"]
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0 if self.headless else 1
        self.process = subprocess.Popen(command, cwd=str(UE_PROJECT.parent), startupinfo=startup,
                                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
        try:
            wait_for_arena(self.process, self.port)
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        (self.output / "launcher.json").write_text(json.dumps({
            "project": str(UE_PROJECT), "python": sys.executable, "port": self.port,
            "owned_pid": self.process.pid if self.process else None,
            "owned_process_stopped": self.process is None or self.process.poll() is not None,
            "error": str(exc) if exc else None,
        }, indent=2), encoding="utf-8")
        return False
