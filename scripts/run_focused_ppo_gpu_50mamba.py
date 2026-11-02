"""PyCharm Run: evaluate the focused fixed-duration PPO GPU baseline."""
from pathlib import Path
import subprocess
import sys

GPU_PYTHON = Path(r"C:\Users\12879\.conda\envs\50mamba\python.exe")


def main():
    if not GPU_PYTHON.is_file():
        raise FileNotFoundError(GPU_PYTHON)
    if Path(sys.executable).resolve() != GPU_PYTHON.resolve():
        return subprocess.run([str(GPU_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]], check=False).returncode
    repro = Path(__file__).resolve().parents[1]/"paper_reproduction"
    sys.path.insert(0, str(repro))
    arguments = list(sys.argv[1:])
    if not any(a == "--task" or a.startswith("--task=") for a in arguments):
        arguments += ["--task", "evaluate"]
    task = next((a.split("=",1)[1] for a in arguments if a.startswith("--task=")), None)
    if task is None:
        task = arguments[arguments.index("--task")+1]
    if task == "evaluate" and not any(a == "--checkpoint" or a.startswith("--checkpoint=") for a in arguments):
        arguments += ["--checkpoint", str(repro/"weights/ppo_focused_gpu_short.zip")]
    from integration.focused_ppo import main as run
    run(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
