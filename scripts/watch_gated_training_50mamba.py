"""PyCharm Run: stream each completed round's actual losses and control metrics."""
from pathlib import Path
import subprocess
import sys

GPU_PYTHON = Path(r"C:\Users\12879\.conda\envs\50mamba\python.exe")


def main():
    if Path(sys.executable).resolve() != GPU_PYTHON.resolve():
        return subprocess.run([str(GPU_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]], check=False).returncode
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_reproduction"))
    from monitoring.round_reporter import main as watch
    arguments = sys.argv[1:]
    if not any(argument in ("--once", "--follow") for argument in arguments):
        arguments = ["--follow", *arguments]
    watch(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
