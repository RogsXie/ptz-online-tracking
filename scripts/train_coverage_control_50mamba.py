"""Run one bounded static control attempt with stratified positions and initial scales."""
from pathlib import Path
import subprocess
import sys

GPU_PYTHON = Path(r'C:\Users\12879\.conda\envs\50mamba\python.exe')


def main():
    if Path(sys.executable).resolve() != GPU_PYTHON.resolve():
        return subprocess.run([str(GPU_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]], check=False).returncode
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'paper_reproduction'))
    from control_coverage_v8.run import main as run
    run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
