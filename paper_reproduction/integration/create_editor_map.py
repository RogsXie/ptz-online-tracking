"""Run with UnrealEditor-Cmd -run=pythonscript, never in policy Python.

Creates a persistent project map with a construction preview arena and named
context assets so opening the project visibly shows the research setup.
"""
import unreal
import json
from pathlib import Path

MAP = "/Game/PTZ/Maps/PTZResearch"
EDITOR = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
LEVEL = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
ACTORS = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def spawn_mesh(name, location, scale, mesh_path, material=None):
    actor = ACTORS.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(*location))
    actor.set_actor_label(name)
    mesh = actor.static_mesh_component
    mesh.set_static_mesh(unreal.load_asset(mesh_path))
    mesh.set_world_scale3d(unreal.Vector(*scale))
    if material:
        mesh.set_material(0, material)
    actor.tags = ["PTZ_EditorContext"]
    # Context is editor-only: runtime uses seeded C++ actors, no duplicate cars.
    actor.set_editor_property("is_editor_only_actor", True)
    actor.set_actor_hidden_in_game(True)
    actor.set_actor_enable_collision(False)
    return actor


def make_material(name, color):
    path = f"/Game/PTZ/Materials/{name}"
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        return unreal.load_asset(path)
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    factory = unreal.MaterialInstanceConstantFactoryNew()
    material = tools.create_asset(name, "/Game/PTZ/Materials", unreal.MaterialInstanceConstant, factory)
    unreal.MaterialEditingLibrary.set_material_instance_parent(
        material, unreal.load_asset("/Engine/BasicShapes/BasicShapeMaterial"))
    unreal.MaterialEditingLibrary.set_material_instance_vector_parameter_value(
        material, "Color", unreal.LinearColor(*color, 1))
    unreal.EditorAssetLibrary.save_asset(path)
    return material


def main():
    if unreal.EditorAssetLibrary.does_asset_exist(MAP):
        raise RuntimeError("Map exists; refuse to overwrite saved user changes")
    unreal.EditorAssetLibrary.make_directory("/Game/PTZ/Maps")
    LEVEL.new_level_from_template(MAP, "/Engine/Maps/Templates/Template_Default")
    arena_class = unreal.load_class(None, "/Script/PTZSim.PTZArena")
    arena = ACTORS.spawn_actor_from_class(arena_class, unreal.Vector(0, 0, 0))
    arena.set_actor_label("PTZ_ResearchArena_RuntimeAndConstructionPreview")
    road = make_material("MI_Road", (.09, .10, .11))
    white = make_material("MI_RoadMark", (.85, .85, .8))
    wall = make_material("MI_Concrete", (.4, .42, .44))
    spawn_mesh("Preview_Road", (2200, 0, 2), (75, 42, .04), "/Engine/BasicShapes/Cube", road)
    for y in (-1500, 0, 1500):
        for x in range(-800, 6500, 600):
            spawn_mesh(f"Preview_Lane_{x}_{y}", (x, y, 5), (3, .09, .02), "/Engine/BasicShapes/Cube", white)
    spawn_mesh("Preview_OcclusionWall", (1400, 500, 290), (3, 4, 5.8), "/Engine/BasicShapes/Cube", wall)
    spawn_mesh("Preview_OcclusionPillar", (1800, -1100, 250), (1.1, 1.1, 5), "/Engine/BasicShapes/Cylinder", wall)
    label = ACTORS.spawn_actor_from_class(unreal.TextRenderActor, unreal.Vector(1000, -900, 500),
                                          unreal.Rotator(0, 180, 0))
    label.set_actor_label("PTZ_ProjectScene_Explanation")
    label.text_render.set_text("PTZSim - UE research platform\nRGB + HAS-CAM / baselines\nRuntime scenarios: paper_reproduction/scenarios\nPreview geometry only; Python controls synchronous time")
    label.text_render.set_world_size(55)
    label.set_editor_property("is_editor_only_actor", True)
    label.tags = ["PTZ_EditorContext"]
    label.set_actor_hidden_in_game(True)
    label.set_actor_enable_collision(False)
    unreal.EditorLevelLibrary.set_level_viewport_camera_info(unreal.Vector(-3000, -4200, 2600),
                                                           unreal.Rotator(-22, 40, 0))
    if not LEVEL.save_current_level():
        raise RuntimeError("Could not save project map (check file locks)")
    unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
    output = Path(unreal.Paths.project_dir()) / "Saved" / "PTZMapCreation.json"
    output.write_text(json.dumps({"map": MAP, "arena": str(arena.get_path_name()),
                                  "saved": True}), encoding="utf-8")
    unreal.log("PTZ_RESEARCH_MAP_SAVED " + MAP)


main()
