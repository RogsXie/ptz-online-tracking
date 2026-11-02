"""PyCharm: right-click this file and choose Run.

Interpreter: D:\\EI\\ptz_ue\\python_env\\Scripts\\python.exe
No Visual Studio session or manually started UE editor is required.
This runs the image-feedback baseline, not the paper's reinforcement learner.
"""
import argparse
import json
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UE_EDITOR = Path(r"D:\UE\UE_5.7\Engine\Binaries\Win64\UnrealEditor.exe")
UE_PROJECT = PROJECT_ROOT / "PTZSim" / "PTZSim.uproject"


def free_port():
    for port in range(9001, 9011):
        try:
            with socket.socket() as candidate:
                candidate.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
    raise RuntimeError("Ports 9001-9010 are busy. Close an unused standalone arena first.")


def wait_for_arena(process, port, timeout=60):
    from unrealcv import Client

    deadline = time.monotonic() + timeout
    expected_path = str(UE_PROJECT.parent).replace("\\", "/").lower()
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"UE exited during startup, exit code {process.returncode}")
        # TCP readiness alone is insufficient: wait until the game arena exists.
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                pass
        except OSError:
            time.sleep(0.3)
            continue
        client = Client(("127.0.0.1", port))
        try:
            client.connect(timeout=2)
            if client.isconnected():
                status = client.request("vget /unrealcv/status", timeout=2)
                if isinstance(status, str) and expected_path in status.replace("\\", "/").lower():
                    reply = client.request("vget /ptz/state", timeout=2)
                    if isinstance(reply, str) and not reply.lower().startswith("error"):
                        if json.loads(reply).get("scene") == "PTZ_Arena_v1":
                            return
        finally:
            client.disconnect()
        time.sleep(0.3)
    raise TimeoutError("UE arena did not become ready within 60 seconds; see this run's ue.log")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true", help="Render without a window for verification")
    parser.add_argument("--fast", action="store_true", help="Disable real-time pacing")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    try:
        from demo_tracking import main as run_demo
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"Missing dependency: {error.name}\n"
            f"In PyCharm, select existing interpreter:\n"
            f"{PROJECT_ROOT / 'python_env' / 'Scripts' / 'python.exe'}"
        ) from error
    for required in [UE_EDITOR, UE_PROJECT,
                     UE_PROJECT.parent / "Binaries" / "Win64" / "UnrealEditor-PTZSim.dll"]:
        if not required.is_file():
            raise FileNotFoundError(f"Required UE file is missing: {required}")

    port = free_port()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = PROJECT_ROOT / "runs" / "pycharm-demo" / timestamp
    output.mkdir(parents=True)
    project_map = UE_PROJECT.parent / "Content/PTZ/Maps/PTZResearch.umap"
    map_name = "/Game/PTZ/Maps/PTZResearch" if project_map.exists() else "/Engine/Maps/Templates/Template_Default"
    command = [str(UE_EDITOR), str(UE_PROJECT), map_name,
               "-game", "-windowed", "-ResX=960", "-ResY=540", "-NoSplash",
               f"-cvport={port}", f"-abslog={output / 'ue.log'}"]
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0 if args.headless else 1
    if args.headless:
        command += ["-RenderOffscreen", "-unattended"]
    print(f"[1/3] Starting UE scene on port {port} ...", flush=True)
    print(f"Python interpreter: {sys.executable}", flush=True)
    process = subprocess.Popen(command, cwd=str(UE_PROJECT.parent), startupinfo=startup,
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    result = {"ue_pid": process.pid, "port": port, "project": str(UE_PROJECT),
              "python": sys.executable, "steps": args.steps, "completed": False}
    try:
        wait_for_arena(process, port)
        print("[2/3] Scene ready. Running RGB detection and PTZ tracking ...", flush=True)
        print("The preview is an HSV/proportional-control baseline, not a learned policy.", flush=True)
        demo_args = ["--steps", str(args.steps), "--seed", str(args.seed),
                     "--port", str(port), "--output", str(output)]
        if not args.fast:
            demo_args += ["--realtime"]
        # Run in this Python process so PyCharm breakpoints work in demo_tracking.py.
        run_demo(demo_args)
        result["completed"] = True
        print(f"[3/3] Demo finished. Annotated video:\n{output / 'tracking-boxed.mp4'}", flush=True)
    finally:
        # Never close the user's pre-existing editor or any other arena process.
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        result["owned_ue_process_stopped"] = process.poll() is not None
        (output / "launcher-results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Preview stopped.", flush=True)
