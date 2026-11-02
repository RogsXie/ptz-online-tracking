"""UE RGB collection and SSD adaptation. GT is allowed only for dataset labels.

Camera aiming here deliberately uses simulator geometry to collect diverse
training crops; the RL environment never calls this sampler or /ptz/view.
"""
import argparse
import json
import math
from pathlib import Path
import random
import time
import cv2
import numpy as np
import torch
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.ops import box_iou
from integration.runtime import REPRO, StandaloneArena, new_run
from integration.transport import RGBTransport
from integration.metrics import iou


def collect(output, counts=(535, 64, 64), headless=True):
    output = Path(output)
    rng = random.Random(20261030)
    splits = json.loads((REPRO / "scenarios/splits.json").read_text(encoding="utf-8-sig"))
    manifest = {"platform": "UE5.7.4 replacement, not author CamSim", "seed": 20261030,
                "labels": "amodal projected full-car bounds, center-occluded targets excluded",
                "camera_sampling": "GT aimed dataset collection only; never used by policy",
                "split_definition": splits,
                "build_sha256": __import__('hashlib').sha256((REPRO.parent / 'PTZSim/Binaries/Win64/UnrealEditor-PTZSim.dll').read_bytes()).hexdigest(),
                "images": []}
    with StandaloneArena(output, headless=headless) as process:
        arena = RGBTransport(process.port)
        try:
            for split, count in zip(("train", "validation", "test"), counts):
                folder = output / split
                folder.mkdir(parents=True, exist_ok=True)
                scenarios = splits[f"{split}_scenarios"]
                seeds = splits["seed_partitions"][split]
                for index in range(count):
                    scenario = scenarios[index % len(scenarios)]
                    seed = seeds[(index * 7) % len(seeds)]
                    arena.scenario(REPRO / "scenarios" / f"{scenario}.json")
                    arena.reset(seed)
                    # Sample real trajectory time without using target information for a policy.
                    for _ in range(index % 7):
                        arena.diagnostics = arena.rpc("vset /ptz/step 0 0 0 1")
                    diagnostic = arena.diagnostics
                    target = diagnostic["targets"][index % len(diagnostic["targets"])]
                    location = target["location_cm"]
                    camera = diagnostic["camera_location"]
                    dx, dy, dz = np.asarray(location) + [0, 0, 90] - np.asarray(camera)
                    yaw = math.degrees(math.atan2(dy, dx))
                    pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
                    fov = rng.choice((8, 12, 18, 25, 40, 60))
                    yaw += rng.uniform(-.25, .25) * fov
                    pitch += rng.uniform(-.10, .10) * fov
                    arena.diagnostics = arena.rpc(f"vset /ptz/view {pitch:.5f} {yaw:.5f} {fov}")
                    rgb = arena.capture()
                    h, w = rgb.shape[:2]
                    boxes, ids, ignored = [], [], []
                    for t in arena.diagnostics["targets"]:
                        if t.get("gt_center_occluded") and t.get("gt_occluder", "").startswith(("StaticMeshActor_", "DefaultPawn")):
                            raise RuntimeError("Unexpected template/preview occluder in labels; audit hidden actor collisions before training")
                        box = np.clip(t.get("gt_bbox_normalized", [0]*4), 0, 1)
                        if not t.get("gt_in_frame") or np.any((box[2:] - box[:2]) * [w, h] < 3):
                            continue
                        if t.get("gt_center_occluded", False):
                            ignored.append(box.tolist())
                            continue
                        boxes.append((box * [w, h, w, h]).tolist())
                        ids.append(t["id"])
                    name = f"{split}/{index:05d}.png"
                    cv2.imwrite(str(output / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    record = {"image": name, "split": split, "seed": seed, "scenario": scenario,
                              "boxes_xyxy_px": boxes, "ids_for_offline_audit": ids,
                              "ignored_center_occluded_normalized": ignored,
                              "diagnostic": arena.diagnostics}
                    manifest["images"].append(record)
                    if index % 25 == 0 or index + 1 == count:
                        print(f"dataset {split} {index + 1}/{count}", flush=True)
                (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        finally:
            arena.close()
    return output / "manifest.json"


class UEDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, split):
        self.root = Path(manifest).parent
        if (self.root / "INVALID_DATASET.json").exists():
            raise ValueError(f"Dataset is marked invalid: {self.root}")
        self.rows = [r for r in json.loads(Path(manifest).read_text(encoding="utf-8"))["images"]
                     if r["split"] == split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        rgb = cv2.cvtColor(cv2.imread(str(self.root / row["image"])), cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255
        boxes = torch.tensor(row["boxes_xyxy_px"], dtype=torch.float32).reshape(-1, 4)
        return image, {"boxes": boxes, "labels": torch.full((len(boxes),), 3, dtype=torch.int64)}


def train(manifest, output, epochs=16, batch_size=8, device="cpu"):
    torch.manual_seed(20261030)
    np.random.seed(20261030)
    torch.set_num_threads(6)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(REPRO / "weights/torch_hub"))
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT).to(device)
    # Preserve pretrained visual features; adapt real SSD classification and box heads.
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)
    dataset = UEDataset(manifest, "train")
    provenance = {"model": "torchvision SSD300 VGG16, COCO initialization, UE head adaptation",
                  "adapted_to_UE": True, "manifest": str(Path(manifest).resolve()),
                  "training_images": len(dataset), "epochs": epochs, "batch_size": batch_size,
                  "device": device, "seed": 20261030, "head_only": True,
                  "optimization": "frozen backbone FP32 features cached once; identical SSD head/loss and anchor matching",
                  "difference_from_paper": "UE images; frozen COCO backbone; Torchvision rather than MMDetection; no author detector weights"}
    logs = []
    started = time.monotonic()
    cached = []
    model.eval()
    # The backbone is frozen and transform deterministic. Cache its FP32
    # features to avoid recomputing VGG for every head-adaptation epoch.
    for index in range(len(dataset)):
        image, target = dataset[index]
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}
        with torch.no_grad():
            transformed, transformed_targets = model.transform([image], [target])
            features = list(model.backbone(transformed.tensors).values())
            anchors = model.anchor_generator(transformed, features)[0]
            target = transformed_targets[0]
            matched = (model.proposal_matcher(box_iou(target["boxes"], anchors))
                       if target["boxes"].numel() else
                       torch.full((len(anchors),), -1, dtype=torch.int64, device=device))
        cached.append(([f[0].detach().cpu() for f in features],
                       {k: v.cpu() for k, v in target.items()}, anchors.cpu(), matched.cpu()))
        if index % 50 == 0 or index + 1 == len(dataset):
            print(f"SSD frozen feature cache {index + 1}/{len(dataset)}", flush=True)
    generator = torch.Generator().manual_seed(20261030)
    model.train()
    model.backbone.eval()
    image, original_target = dataset[0]
    with torch.no_grad():
        direct = model([image.to(device)], [{k: v.to(device) for k, v in original_target.items()}])
        first = cached[0]
        cached_loss = model.compute_loss(
            [{k: v.to(device) for k, v in first[1].items()}],
            model.head([f[None].to(device) for f in first[0]]),
            [first[2].to(device)], [first[3].to(device)])
    difference = max(abs(float(direct[k] - cached_loss[k])) for k in direct)
    if difference > 1e-5:
        raise RuntimeError(f"Cached SSD loss differs from original forward: {difference}")
    provenance["cached_loss_max_difference_from_direct_forward"] = difference
    print(f"SSD cached/direct loss max difference {difference}", flush=True)
    with (output / "training.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(epochs):
            model.train()
            model.backbone.eval()
            losses = []
            indices = torch.randperm(len(cached), generator=generator).tolist()
            for offset in range(0, len(indices), batch_size):
                rows = [cached[i] for i in indices[offset:offset + batch_size]]
                features = [torch.stack([row[0][level] for row in rows]).to(device)
                            for level in range(len(rows[0][0]))]
                targets = [{k: v.to(device) for k, v in row[1].items()} for row in rows]
                anchors = [row[2].to(device) for row in rows]
                matches = [row[3].to(device) for row in rows]
                values = model.compute_loss(targets, model.head(features), anchors, matches)
                loss = sum(values.values())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite detector loss {values}")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), 10)
                optimizer.step()
                losses.append(float(loss.detach()))
            record = {"epoch": epoch + 1, "loss": float(np.mean(losses)),
                      "wall_seconds": time.monotonic() - started}
            logs.append(record)
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(record, flush=True)
            payload = {"model": model.cpu().state_dict(), "provenance": provenance,
                       "epochs_completed": epoch + 1, "training_log": logs}
            torch.save(payload, output / "ssd_ue.pt")
            model.to(device)
    (output / "config.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return output / "ssd_ue.pt"


def evaluate(manifest, checkpoint, split, output):
    from integration.detector import SSDDetector
    detector = SSDDetector(checkpoint, threshold=.05)
    data = UEDataset(manifest, split)
    scores, positives, matches = [], 0, []
    for index, row in enumerate(data.rows):
        rgb = cv2.cvtColor(cv2.imread(str(data.root / row["image"])), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        gt = [np.array(b) / [w, h, w, h] for b in row["boxes_xyxy_px"]]
        ignore = row["ignored_center_occluded_normalized"]
        positives += len(gt)
        used = set()
        for detection in detector(rgb):
            candidates = sorted(((iou(detection.bbox, box), j) for j, box in enumerate(gt)), reverse=True)
            best = next(((v, j) for v, j in candidates if j not in used), (0, -1))
            true = best[0] >= .5
            if true:
                used.add(best[1])
            elif any(iou(detection.bbox, b) >= .3 for b in ignore):
                continue
            scores.append((detection.confidence, int(true)))
        if index % 20 == 0:
            print(f"detector evaluation {split} {index}/{len(data)}", flush=True)
    scores.sort(reverse=True)
    flags = np.array([v for _, v in scores])
    tp = np.cumsum(flags)
    precision = tp / np.arange(1, len(flags) + 1) if len(flags) else np.array([])
    recall = tp / max(1, positives)
    ap = float(np.mean([max(precision[recall >= r], default=0) for r in np.linspace(0, 1, 101)]))
    chosen = [v for c, v in scores if c >= .15]
    result = {"split": split, "images": len(data), "gt_boxes": positives, "AP50_101_point": ap,
              "precision_at_015": sum(chosen) / max(1, len(chosen)),
              "recall_at_015": sum(chosen) / max(1, positives),
              "checkpoint": str(checkpoint), "provenance": detector.provenance,
              "label_limit": "full-car projected amodal bounds; center-occluded instances ignored, not segmentation or real-world accuracy"}
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["collect", "train", "evaluate"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--counts", type=int, nargs=3, default=[535, 64, 64])
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    args = parser.parse_args()
    if args.task == "collect":
        collect(args.output or new_run("detector-dataset"), args.counts)
    elif args.task == "train":
        train(args.manifest, args.output or new_run("detector-training"), args.epochs)
    else:
        print(evaluate(args.manifest, args.checkpoint, args.split, args.output))


if __name__ == "__main__":
    main()
