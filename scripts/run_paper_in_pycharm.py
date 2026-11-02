"""PyCharm right click Run: trained HAS-CAM short-run checkpoint on held-out UE scenes.

Interpreter D:\\EI\\ptz_ue\\python_env\\Scripts\\python.exe.
This checkpoint is a short verification run, not a converged paper result.
For training use --task train --episodes 12 --fast; evaluation uses no exploration.
Old run_in_pycharm.py remains the explicitly labelled HSV/proportional baseline.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_reproduction"))
from integration.experiment import main


if __name__ == "__main__":
    main()
