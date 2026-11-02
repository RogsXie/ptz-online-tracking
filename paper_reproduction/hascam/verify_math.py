"""Independent numerical checks; these are not UE training/convergence results.

Run in PyCharm or with python .../hascam/verify_math.py. The source-equivalence
check extracts only class definitions from the pinned local author repository.
"""
from pathlib import Path
import argparse
import ast
import json
import math
import sys
import tempfile
import time
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from hascam import Detection, StateBuilder, VPEReward, HASCamAgent, PDQNConfig, CFFF
from hascam.config import ACTION_NAMES, ACTION_RATES, epsilon
from hascam.state_reward import LOST_STATE
from hascam.reid import FastReIDExtractor


def definitions(path, names, namespace):
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    assert len(nodes) == len(names), (path, names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def source_backbone_equivalence(extractor):
    source = ROOT / "reference" / "HAS-CAM" / "fastreid"
    scope = {"torch": torch, "nn": nn, "math": math,
             "get_norm": lambda name, n, **kwargs: nn.BatchNorm2d(n)}
    definitions(source / "layers" / "batch_norm.py", {"IBN"}, scope)
    definitions(source / "layers" / "non_local.py", {"Non_local"}, scope)
    definitions(source / "modeling" / "backbones" / "resnet.py", {"BasicBlock", "Bottleneck", "ResNet"}, scope)
    original = scope["ResNet"](1, "BN", True, False, True, scope["Bottleneck"], [3,4,6,3], [0,2,3,0]).eval()
    original.load_state_dict(extractor.model.backbone.state_dict(), strict=True)
    generator = torch.Generator().manual_seed(8841)
    values = torch.randn(1,3,256,256, generator=generator)
    with torch.inference_mode():
        expected = original(values)
        actual = extractor.model.backbone(values)
    max_difference = float((expected-actual).abs().max())
    assert max_difference < 1e-6, max_difference
    return {"max_abs_difference": max_difference, "shape": list(actual.shape),
            "source": str(source / "modeling" / "backbones" / "resnet.py")}


def run(output):
    torch.set_num_threads(4)
    checks = []

    def check(name, fn):
        start = time.perf_counter()
        try:
            details = fn()
            checks.append({"name": name, "passed": True, "seconds": time.perf_counter()-start,
                           "details": details})
        except Exception as error:
            checks.append({"name": name, "passed": False, "seconds": time.perf_counter()-start,
                           "error": f"{type(error).__name__}: {error}"})

    def state_table():
        b = StateBuilder()
        one = Detection((.4,.4,.6,.6), .8)
        zero = b.build(one, reset=True)
        np.testing.assert_allclose(zero, [0,0,.04,.8,0,0,0,0], atol=1e-7)
        two = b.build(Detection((.5,.35,.7,.55), .7), (3,.4))
        np.testing.assert_allclose(two, [.2,-.1,.04,.7,3/7,.4,.2,-.1], atol=1e-7)
        np.testing.assert_array_equal(b.build(None, (4,.3)), LOST_STATE)
        np.testing.assert_allclose(b.build(one)[6:], 0)
        return {"reset_state": zero.tolist(), "moved_state": two.tolist(),
                "inputs": "Detection bbox/confidence and preceding action only; no camera/GT fields"}

    check("Table1 reset, trend, history normalization and lost sentinels", state_table)

    def actions():
        assert ACTION_NAMES == ("Stay", "Down", "Up", "Right", "Left", "ZoomIn", "ZoomOut")
        assert ACTION_RATES[3] == (1.,0.,0.) and ACTION_RATES[1] == (0.,-1.,0.)
        np.testing.assert_allclose([epsilon(0),epsilon(150),epsilon(300),epsilon(4060)], [.9,.5,.1,.1])
        a = HASCamAgent(seed=301)
        parameters = np.array([a.select_action(np.zeros(8,np.float32), True)[1] for _ in range(200)])
        assert np.isfinite(parameters).all() and parameters.min() >= .04 and parameters.max() <= 1
        assert len(np.unique(parameters)) > 100
        return {"action_names": list(ACTION_NAMES), "duration_min": float(parameters.min()),
                "duration_max": float(parameters.max()), "unique_continuous_durations": len(np.unique(parameters))}

    check("Eq1 action mapping, Eq2 continuous bounds and Eq13 epsilon", actions)

    def reward_values():
        # 5%-area centered confident box: literal reward deliberately differs.
        # Use just above .05: the printed literal area formula is discontinuous
        # at .05, so rounding an exactly 5% square can choose its other branch.
        half = math.sqrt(.05000001)/2
        good = Detection((.5-half,.5-half,.5+half,.5+half), 1.)
        off = Detection((.75,.6,.95,.85), .5)
        literal, actual = VPEReward("paper_literal"), VPEReward("author_code")
        r_literal = literal.step(good)["reward"]
        r_actual = actual.step(good)["reward"]
        assert abs(r_literal+2) < 1e-6 and abs(r_actual) < 1e-6
        assert VPEReward().step(None)["reward"] == -4
        assert VPEReward.terminal_adjustment(True,127) == 40
        assert VPEReward.terminal_adjustment(True,128) == -40
        for a in [1e-7,.001,.049999,.05,.050001,.2,1.]:
            half = math.sqrt(a)/2
            assert math.isfinite(VPEReward().step(Detection((.5-half,.5-half,.5+half,.5+half),.5))["reward"])
        return {"centered_confident_5pct_author_reward": r_actual,
                "same_paper_literal_reward": r_literal, "lost_reward": -4,
                "at_step128_terminal_adjustment": -40}

    check("Eqs5-12 boundaries and explicit printed/source reward conflict", reward_values)

    def cfff_fixture():
        # Features are exact test fixtures, not a replacement ReID model.
        def fixture(rgb, detection):
            return np.array([1.,0.,0.]) if detection.label == "A" else np.array([0.,1.,0.])
        c = CFFF(fixture, max_memory=1)
        a = Detection((.1,.2,.2,.3), .91, label="A")
        b = Detection((.7,.2,.8,.3), .99, label="B")
        assert c.choose(None,[a],count_step=False) == a and c.confidence_streak == 0
        for step in range(10):
            assert c.choose(None,[b,a]) == a
        assert c.last_event["completed"] and c.completed_count == 1
        assert c.choose(None,[a,b]) == b and c.last_event["new_target"]
        assert c.choose(None,[]) is None and c.confidence_streak == 0
        for step in range(10):
            assert c.choose(None,[b]) == b
        assert c.completed_count == 2 and len(c.monitored) == 1
        # Identical appearance is correctly exposed as a known limitation.
        assert c.choose(None,[Detection((.2,.4,.3,.5),.99,label="B")]) is None
        return {"completion_count": c.completed_count, "memory_fifo_size": len(c.monitored),
                "identical_appearance_collapse_observed": True,
                "fixtures_only": True}

    check("Algorithm1 active matching, monitored exclusion, streak, switch and FIFO", cfff_fixture)

    agent = HASCamAgent(seed=31)

    def update_checkpoint():
        rng = np.random.default_rng(29)
        # Synthetic transitions check numerical mechanics, not task learning.
        for index in range(640):
            s = rng.uniform(-1,1,size=8).astype(np.float32)
            ns = rng.uniform(-1,1,size=8).astype(np.float32)
            action = agent.select_action(s)
            agent.observe(s,action,float(rng.normal()),ns,index%17==0)
        actor_before = deepcopy_state(agent.actor_net)
        param_before = deepcopy_state(agent.param_net)
        actor_target_before = deepcopy_state(agent.actor_target)
        last = None
        for index in range(100):
            last = agent.update()
            assert all(math.isfinite(last[key]) for key in ["q_loss","parameter_objective","parameter_grad_norm"])
            if index == 0:
                assert any(not torch.equal(actor_before[k],v) for k,v in agent.actor_net.state_dict().items())
                assert any(not torch.equal(param_before[k],v) for k,v in agent.param_net.state_dict().items())
                assert all(torch.equal(actor_target_before[k],v) for k,v in agent.actor_target.state_dict().items())
        assert last["target_synchronised"]
        assert all(torch.equal(v,agent.actor_target.state_dict()[k]) for k,v in agent.actor_net.state_dict().items())
        assert all(torch.equal(v,agent.param_net_target.state_dict()[k]) for k,v in agent.param_net.state_dict().items())
        with tempfile.TemporaryDirectory(dir=Path(output).parent) as tmp:
            checkpoint = Path(tmp)/"numerical-checkpoint.pt"
            agent.save(checkpoint, {"purpose":"numerical_only"})
            restored = HASCamAgent.load(checkpoint)
        fixed = np.linspace(-1,1,8,dtype=np.float32)
        expected = agent.select_action(fixed,False)
        actual = restored.select_action(fixed,False)
        assert expected == actual and len(restored.memory) == 640
        assert agent.select_action(fixed,True) == restored.select_action(fixed,True)
        old_metrics,new_metrics = agent.update(),restored.update()
        np.testing.assert_allclose(old_metrics["q_loss"],new_metrics["q_loss"],rtol=0,atol=0)
        return {"synthetic_transitions":640,"optimisations":100,"target_copy_on_update100":True,
                "checkpoint_weights_replay_optimizer_rng_roundtrip":True,"greedy_action":list(actual),
                "last_update":last,"claim":"numerical mechanics only, no policy convergence claim"}

    check("Uniform replay, both parameter updates, terminal mask, hard sync and checkpoint resume", update_checkpoint)

    def terminal_mask():
        losses = []
        for done, reward, expected in ((True,3.,2.5),(False,0.,89.5)):
            a = HASCamAgent(PDQNConfig(batch_size=4),seed=91)
            with torch.no_grad():
                for value in a.actor_net.parameters():
                    value.zero_()
                for value in a.actor_target.parameters():
                    value.zero_()
                a.actor_target.layers[-1].bias.fill_(100.)
            state = np.zeros(8,np.float32)
            for _ in range(4):
                a.observe(state,(0,.5),reward,state,done,all_parameters=np.full(7,.5,np.float32))
            metrics = a.update()
            assert abs(metrics["q_loss"]-expected) < 1e-6, metrics
            losses.append(metrics["q_loss"])
        return {"terminal_huber_loss_expected_2_5":losses[0],
                "nonterminal_huber_loss_expected_89_5":losses[1],
                "next_target_q":100,"gamma":.9}

    check("Independent Bellman target oracle: terminal masks bootstrap, live transition discounts it",terminal_mask)

    def invert_bounds():
        values = torch.tensor([[.04,1.,.04,1.,.52,.52,.52]])
        gradients = torch.tensor([[-1.,1.,1.,-1.,1.,-1.,0.]])
        inverted = agent.invert_gradients(gradients,values)
        np.testing.assert_allclose(inverted, [[0,0,1,-1,.5,-.5,0]],atol=1e-6)
        return {"inverted_gradients":inverted.tolist()}

    check("P-DQN inverting gradients vanish outward at duration bounds", invert_bounds)

    extractor = None
    def real_reid():
        nonlocal extractor
        extractor = FastReIDExtractor()
        rgb = np.zeros((540,960,3),np.uint8)
        rgb[100:350,200:700] = [180,30,20]
        feature = extractor(rgb,Detection((200/960,100/540,700/960,350/540),.8))
        assert feature.shape == (2048,) and np.isfinite(feature).all()
        np.testing.assert_allclose(np.linalg.norm(feature),1,atol=1e-6)
        return extractor.metadata()

    check("Official VeRi weights strict inference load, actual 2048-D feature and norm", real_reid)
    if extractor is not None:
        check("Portable ReID backbone equals pinned author FastReID source", lambda:source_backbone_equivalence(extractor))
    result = {"kind":"algorithm_numerical_verification", "policy_training":False,
              "python":sys.executable,"torch":torch.__version__,"device":"cpu",
              "checks":checks,"passed":all(c["passed"] for c in checks),
              "passed_count":sum(c["passed"] for c in checks),"check_count":len(checks)}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result["passed"] else 1


def deepcopy_state(model):
    return {key:value.detach().clone() for key,value in model.state_dict().items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path(__file__).parent/"verification"/"math-results.json")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    raise SystemExit(run(args.output))
