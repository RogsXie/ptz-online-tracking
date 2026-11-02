"""New declared holdout block; identical static distribution and quality gates."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import random
from control_curriculum import scenes as original
from control_curriculum.scenes import stage_spec, load_protocol
from control_curriculum.gates import assess_episode

HERE = Path(__file__).parent
MANIFEST = json.loads((HERE / "evaluation-manifest.json").read_text(encoding="utf-8"))


def manifest_hash():
    return hashlib.sha256((HERE / "evaluation-manifest.json").read_bytes()).hexdigest()


def cases(stage, split, count=None, offset=0, allow_final_test=False):
    if stage != "static_center" or split not in ("train", "validation"):
        raise ValueError("v6 limited to static development/validation; final test reserved")
    if split == "train":
        return original.cases(stage, split, count, offset)
    if offset or count not in (None,40):
        raise ValueError("Entire declared40-case holdout is required")
    start, end = MANIFEST["validation_seeds"]
    return [{"stage":stage,"scenario_name":"control_static_center","split":split,
             "case_index":seed-510000,"seed":seed} for seed in range(start,end+1)]


def resolve(stage, split, seed):
    if stage != "static_center" or split not in ("train", "validation"):
        raise ValueError("Static development/validation only")
    if split == "train":
        return original.resolve(stage, split, seed)
    start, end = MANIFEST["validation_seeds"]
    if not start <= seed <= end:
        raise ValueError("Validation seed outside declared new block")
    # Unchanged template, then reproduce original generator using the NEW seed.
    # No offset is obtained from target geometry or rendered observations.
    config = deepcopy(original.resolve(stage, "validation", 510000))
    rng = random.Random(seed)
    ordinal = seed-510000
    yaw_sign,pitch_sign = [(-1,-1),(-1,1),(1,-1),(1,1)][ordinal%4]
    config["targets"][0]["path"]["origin_cm"] = [rng.uniform(4800,5200),rng.uniform(-25,25),8.]
    config["camera"]["rotation_deg"] = [-6.+pitch_sign*rng.uniform(1.,3.2),yaw_sign*rng.uniform(1.2,4.),0.]
    config["control_protocol"]["seed"] = seed
    config["control_protocol"]["evaluation_manifest_sha256"] = manifest_hash()
    return config


def materialize(stage, split, seed, output_dir):
    config = resolve(stage, split, seed)
    path = Path(output_dir) / f"control_{stage}-{split}-seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    value = json.dumps(config, ensure_ascii=False, indent=2)+"\n"
    if path.exists() and path.read_text(encoding="utf-8-sig") != value:
        raise FileExistsError("Refusing to replace a different scenario")
    if not path.exists():
        path.write_text(value,encoding="utf-8")
    return path


def gate_aggregate(stage, episodes):
    spec = stage_spec(stage)
    expected = {case["seed"] for case in cases(stage,"validation")}
    protocol_sha = hashlib.sha256((Path(original.__file__).parent/"protocol.json").read_bytes()).hexdigest()
    if protocol_sha != MANIFEST["quality_protocol_sha256"]:
        raise ValueError("Quality protocol changed")
    seeds = [row.get("seed") for row in episodes]
    passed = sum(row.get("passed") is True for row in episodes)
    rate = passed/len(episodes) if episodes else 0.
    quadrants = {str(i):sum(row.get("passed") is True and row.get("seed") in expected
                           and row["seed"]%4==i for row in episodes) for i in range(4)}
    checks = {"complete_expected_validation_seeds":set(seeds)==expected and len(seeds)==40,
        "no_duplicate_seeds":len(seeds)==len(set(seeds)),
        "stage_matches":all(row.get("stage")==stage for row in episodes),
        "episode_protocol_matches":all(row.get("protocol_sha256")==protocol_sha for row in episodes),
        "success_rate":rate>=spec["validation_success_rate_min"],
        "balanced_quadrants":min(quadrants.values())>=spec["validation_quadrants_min_pass_each"]}
    return {"stage":stage,"passed":all(checks.values()),"advance_allowed":all(checks.values()),
        "checks":checks,"failed_checks":[k for k,v in checks.items() if not v],
        "episodes":len(episodes),"expected_episodes":40,"passed_episodes":passed,
        "success_rate":rate,"minimum_success_rate":spec["validation_success_rate_min"],
        "quadrant_passes":quadrants,"protocol_sha256":protocol_sha,
        "evaluation_manifest_sha256":manifest_hash(),"expected_seeds":sorted(expected),
        "progression_note":"Static stage only; no automatic advance"}
