"""Explicit bounded frozen precision-band joint-development entry."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'paper_reproduction'))
from control_precision_v14.run import main
if __name__=='__main__':main()
