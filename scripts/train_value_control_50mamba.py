"""Launch the single preregistered v9 fixed-pulse control trial."""
from pathlib import Path
import sys

GPU_PYTHON=Path(r'C:\Users\12879\.conda\envs\50mamba\python.exe')

if __name__=='__main__':
    if Path(sys.executable).resolve()!=GPU_PYTHON.resolve():
        raise RuntimeError('Use the verified50mamba interpreter directly')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'paper_reproduction'))
    from control_value_v9.run import main
    main()
