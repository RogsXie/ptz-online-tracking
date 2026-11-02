"""Right-click Run: use the existing 50mamba CUDA environment for HAS-CAM.

Defaults to GPU evaluation. For training pass --task train --episodes 300
--curriculum-ramp 300 --headless --fast. Results use the normal new-run folder.
"""
from pathlib import Path
import subprocess
import sys


GPU_PYTHON = Path(r"C:\Users\12879\.conda\envs\50mamba\python.exe")


def main():
    if not GPU_PYTHON.is_file():
        raise FileNotFoundError(f"50mamba interpreter is missing: {GPU_PYTHON}")
    if Path(sys.executable).resolve() != GPU_PYTHON.resolve():
        print(f"Using CUDA interpreter: {GPU_PYTHON}", flush=True)
        return subprocess.run([str(GPU_PYTHON), str(Path(__file__).resolve()),
                               *sys.argv[1:]], check=False).returncode
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_reproduction"))
    from integration.experiment import main as run_experiment
    arguments = sys.argv[1:]
    if not any(value == "--device" or value.startswith("--device=") for value in arguments):
        arguments = [*arguments, "--device", "cuda"]
    run_experiment(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
