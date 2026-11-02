"""Load original code, never a reimplementation of its learning equations."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

REPRO = Path(__file__).resolve().parents[1]
REFERENCE = REPRO / "reference/HAS-CAM"
COMMIT = "9173676fb636140ffa4145e101631d3f30634267"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source():
    manifest = json.loads((REPRO / "docs/source_manifest.json").read_text())
    checked = {}
    for name, expected in manifest["files"].items():
        actual = digest(REFERENCE / name)
        if actual != expected:
            raise RuntimeError(f"Pinned author source differs: {name}")
        checked[name] = actual
    if manifest["commit"] != COMMIT:
        raise RuntimeError("Author revision changed")
    checked["utils.py"] = digest(REFERENCE / "utils.py")
    return checked


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, REFERENCE / filename)
    loaded = importlib.util.module_from_spec(spec)
    # Do not write pycache into the immutable upstream checkout.
    exec(compile((REFERENCE / filename).read_bytes(), str(REFERENCE / filename), "exec"), loaded.__dict__)
    return loaded


def load_core():
    verify_source()
    model = module("_pinned_hascam_model", "model.py")
    memory = module("_pinned_hascam_memory", "memory.py")
    saved = {name: sys.modules.get(name) for name in ("model", "memory")}
    try:
        sys.modules.update(model=model, memory=memory)
        agent = module("_pinned_hascam_agent", "agent.py")
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    utils = module("_pinned_hascam_utils", "utils.py")
    namespace = {}
    tree = ast.parse((REFERENCE / "main.py").read_text(encoding="utf-8"))
    epsilon = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "epsilon")
    exec(compile(ast.Module(body=[epsilon], type_ignores=[]), str(REFERENCE / "main.py"), "exec"), namespace)
    return agent.PDQNAgent, utils.pad_action, namespace["epsilon"]


def environment_class(namespace):
    """Compile the complete, unedited class; replace only module I/O bindings.

    Avoid executing the source module's top-level UDP socket, obsolete detector
    imports and FastReID training framework. Every class method AST is original.
    """
    path = REFERENCE / "rl_control/UnityCameraEnvironment_exp.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UnityCameraControlEnv")
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["UnityCameraControlEnv"]


def recorded_update(agent):
    """Read original update's return-frame locals without changing its math/RNG."""
    record = {}
    code = agent.update.__func__.__code__
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if event == "return" and frame.f_code is code and "loss_actor" in frame.f_locals:
            values = frame.f_locals
            record.update(q_loss=float(values["loss_actor"].detach()),
                          parameter_objective=float(values["param_loss"].detach()),
                          parameter_surrogate=float(values["out"].detach().sum()),
                          update_count=agent.update_count,
                          replay_size=len(agent.memory),
                          target_synchronised=agent.update_count % 100 == 0)
            import torch
            record["parameter_grad_norm_after_clip"] = float(torch.linalg.vector_norm(torch.stack([
                p.grad.detach().norm() for p in agent.param_net.parameters() if p.grad is not None])))
        if previous is not None:
            previous(frame, event, arg)

    try:
        sys.setprofile(observe)
        agent.update()
    finally:
        sys.setprofile(previous)
    return record or None


def create_agent(seed=1, device="cuda"):
    import random
    import gymnasium as gym
    import numpy as np
    import torch
    klass, _, _ = load_core()
    # Reproducible initialization supplements main.py's incomplete seed wiring;
    # default Linear initialization and all original sampling methods remain.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    action_space = gym.spaces.Tuple((gym.spaces.Discrete(7), *[
        gym.spaces.Box(np.array([.3], np.float32), np.array([5.], np.float32)) for _ in range(7)]))
    return klass(gym.spaces.Discrete(8), action_space, batch_size=512, gamma=.9, seed=seed,
                 actor_kwargs={"hidden_layers": [128,256,128,64]},
                 param_net_kwargs={"hidden_layers": [128,256,128,64]}, device=device)
