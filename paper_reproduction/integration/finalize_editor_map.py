"""Hide only our newly-created editor context in uncooked standalone games."""
import unreal
import json
from pathlib import Path

level = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
level.load_level("/Game/PTZ/Maps/PTZResearch")
labels = []
for actor in actors.get_all_level_actors():
    if "PTZ_EditorContext" in [str(t) for t in actor.tags] or actor.get_actor_label() == "PTZ_ProjectScene_Explanation":
        actor.set_actor_hidden_in_game(True)
        actor.set_actor_enable_collision(False)
        labels.append(actor.get_actor_label())
if not level.save_current_level():
    raise RuntimeError("Could not save PTZResearch map; another process may hold it open")
(Path(unreal.Paths.project_saved_dir()) / "PTZMapFinalized.json").write_text(
    json.dumps({"map": "/Game/PTZ/Maps/PTZResearch", "hidden_game_context": labels}), encoding="utf-8")
unreal.log("PTZ_MAP_FINALIZED " + str(len(labels)))
