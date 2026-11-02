"""Isolated speed revision; all released source and checkpoints stay untouched."""
from pathlib import Path
import sys
REV=Path(__file__).resolve().parent
LEGACY=REV.parent
sys.path.insert(0,str(LEGACY))
import bootstrap
REPRO=bootstrap.REPRO
sys.dont_write_bytecode=True

def configure_precision_project():
    import importlib
    import integration.runtime as runtime
    project=REV/'UE/PTZSim/PTZSim.uproject'
    assert project.is_file() and (project.parent/'Binaries/Win64/UnrealEditor-PTZSim.dll').is_file()
    runtime.UE_PROJECT=project
    # The existing readiness checker has its own imported project constant.
    importlib.import_module(runtime.wait_for_arena.__module__).UE_PROJECT=project
    return project
