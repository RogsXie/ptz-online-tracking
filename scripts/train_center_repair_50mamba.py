"""Launch the bounded Phase A of the authorized centering/zoom repair."""
from pathlib import Path
import sys

if __name__=='__main__':
    if Path(sys.executable).resolve()!=Path('C:/Users/12879/.conda/envs/50mamba/python.exe').resolve():
        raise RuntimeError('Use50mamba Python')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'paper_reproduction'))
    from control_center_v12a.run import main
    main()
