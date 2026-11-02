"""Explicit entry for the bounded learned RGB model experiment."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'paper_reproduction'))
from control_model_v13.run import main
if __name__=='__main__':main()
