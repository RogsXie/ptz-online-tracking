"""Launch one bounded seven-action active-zoom trial."""
from pathlib import Path
import sys

if __name__ == '__main__':
    if Path(sys.executable).resolve() != Path('C:/Users/12879/.conda/envs/50mamba/python.exe').resolve():
        raise RuntimeError('Use the verified 50mamba interpreter directly')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'paper_reproduction'))
    from control_zoom_v11.run import main
    main()
