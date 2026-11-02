"""PyCharm Run: focused five-family GPU demo. Pass --task train --headless to train."""
from pathlib import Path
import subprocess
import sys

GPU_PYTHON = Path(r"C:\Users\12879\.conda\envs\50mamba\python.exe")


def default_arguments(arguments):
    arguments = list(arguments)
    task = "demo"
    for index, value in enumerate(arguments):
        if value == "--task":
            task = arguments[index+1] if index+1 < len(arguments) else None
        elif value.startswith("--task="):
            task = value.split("=", 1)[1]
    present = lambda name: any(value == name or value.startswith(name+"=") for value in arguments)
    if task == "demo":
        if not present("--task"):
            arguments += ["--task", "demo"]
        if not present("--episodes"):
            arguments += ["--episodes", "5"]
        if not present("--eval-steps"):
            arguments += ["--eval-steps", "128"]
    return arguments


def main():
    if not GPU_PYTHON.is_file():
        raise FileNotFoundError(GPU_PYTHON)
    if Path(sys.executable).resolve() != GPU_PYTHON.resolve():
        return subprocess.run([str(GPU_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]], check=False).returncode
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"paper_reproduction"))
    from integration.focused_training import main as run
    arguments = default_arguments(sys.argv[1:])
    run(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
