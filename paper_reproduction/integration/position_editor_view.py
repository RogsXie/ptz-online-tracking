"""Save the project view using -ExecutePythonScript, then reopen normally.

UE's script executor closes this helper editor after completion. The normal
Start-PTZSim.ps1 entry opens the persisted view as an interactive editor.
"""
import json
from pathlib import Path
import unreal

level = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
if not level.load_level("/Game/PTZ/Maps/PTZResearch"):
    raise RuntimeError("Could not open the saved project map")
editor = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
# Python's Rotator positional order is roll/pitch/yaw; always name components.
editor.set_level_viewport_camera_info(
    unreal.Vector(-1400, -2600, 2200), unreal.Rotator(pitch=-26.4, yaw=35.8, roll=0.))
actual = editor.get_level_viewport_camera_info()
location = next((value for value in actual if isinstance(value, unreal.Vector)), None)
rotation = next((value for value in actual if isinstance(value, unreal.Rotator)), None)
if location is None or rotation is None:
    raise RuntimeError(f"Viewport API did not return an actual camera transform: {actual}")
if abs(rotation.pitch + 26.4) > .001 or abs(rotation.yaw - 35.8) > .001 or abs(rotation.roll) > .001:
    raise RuntimeError(f"Actual viewport rotation differs: {rotation}")
if not level.save_current_level():
    raise RuntimeError("Viewport/map save failed; close standalone readers before saving")
result = {"map": "/Game/PTZ/Maps/PTZResearch", "saved": True,
          "viewport_location": [location.x, location.y, location.z],
          "viewport_rotation_pitch_yaw_roll": [rotation.pitch, rotation.yaw, rotation.roll],
          "arena_count": sum(actor.get_class().get_name() == "PTZArena" for actor in actors.get_all_level_actors()),
          "purpose": "editor view only; no Play, no policy, no simulator truth following"}
(Path(unreal.Paths.project_saved_dir()) / "PTZEditorView.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
unreal.log("PTZ_EDITOR_VIEW_READY")
