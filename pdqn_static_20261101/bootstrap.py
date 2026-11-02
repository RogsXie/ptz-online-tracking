"""Independent experiment; historical code and checkpoints stay read-only."""
from pathlib import Path
import os
import sys

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent / 'paper_reproduction'
sys.path.insert(0, str(REPRO))
sys.dont_write_bytecode = True
os.environ['YOLO_CONFIG_DIR'] = str(HERE / 'runtime/ultralytics')
os.environ['YOLO_AUTOINSTALL'] = 'false'

