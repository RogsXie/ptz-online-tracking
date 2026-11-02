"""Verify a running PTZSim Play-in-Editor session over local UnrealCV.

Uses an independent camera, checks parameter readback and image changes, and
restores that camera's original rotation, FOV, and resolution.
This is an interface check, not a tracking or reinforcement-learning experiment.
"""

import argparse
import io
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat
from unrealcv import Client


def text_response(client, command):
    reply = client.request(command, timeout=10)
    if isinstance(reply, bytes):
        reply = reply.decode("utf-8")
    if not isinstance(reply, str) or reply.lower().startswith("error"):
        raise RuntimeError(f"{command}: {reply!r}")
    return reply


def vector(client, command, size):
    reply = text_response(client, command)
    values = [float(value) for value in reply.split()]
    if len(values) != size or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"Unexpected response to {command}: {reply!r}")
    return values


def capture(client, camera, path):
    data = client.request(f"vget /camera/{camera}/lit png", timeout=20)
    if not isinstance(data, bytes):
        raise RuntimeError(f"Expected PNG bytes, got {str(data)[:200]}")
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.format != "PNG":
            raise RuntimeError("Image response is not PNG")
        stats = ImageStat.Stat(image.convert("RGB"))
        result = {
            "file": str(path),
            "width": image.width,
            "height": image.height,
            "rgb_standard_deviation": stats.stddev,
        }
        if max(stats.stddev) < 1:
            raise RuntimeError("Captured image is nearly uniform")
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    path.write_bytes(data)
    return result, pixels


def lower_half_difference(before, after):
    """Use the static template floor, excluding most moving sky/cloud pixels."""
    start = before.shape[0] // 2
    return float(np.abs(before[start:] - after[start:]).mean())


def close_angle(actual, expected):
    return abs((actual - expected + 180) % 360 - 180) < 0.1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--camera", type=int, default=1,
                        help="Independent camera ID; camera 0 is the player view")
    parser.add_argument("--settle-seconds", type=float, default=0.5,
                        help="Allow a game/render tick after each camera action")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "verification")
    args = parser.parse_args()
    if args.camera < 1 or args.settle_seconds < 0:
        parser.error("Use an independent camera ID >= 1 and a nonnegative delay")
    args.output.mkdir(parents=True, exist_ok=True)
    client = Client(("127.0.0.1", args.port))
    client.connect(timeout=5)
    if not client.isconnected():
        raise RuntimeError("Cannot connect to UnrealCV; open PTZSim and start Play first")
    original_rotation = None
    original_fov = None
    original_size = None
    result = {"endpoint": f"127.0.0.1:{args.port}", "camera_id": args.camera,
              "checks": {}, "captures": [], "image_effects": {}}
    camera_path = f"/camera/{args.camera}"

    def take(name):
        metadata, pixels = capture(client, args.camera, args.output / f"{name}.png")
        result["captures"].append(metadata)
        return pixels

    try:
        result["server_status"] = text_response(client, "vget /unrealcv/status")
        cameras = text_response(client, "vget /cameras").split()
        if args.camera == len(cameras):
            result["spawned_actor"] = text_response(client, "vset /cameras/spawn")
            # A fixed viewpoint above the floor of the blank template map.
            text_response(client, f"vset {camera_path}/location 0 0 200")
            text_response(client, f"vset {camera_path}/rotation -15 0 0")
            time.sleep(args.settle_seconds)
        elif args.camera > len(cameras):
            raise RuntimeError(f"Camera {args.camera} does not exist; available: {cameras}")
        result["cameras"] = text_response(client, "vget /cameras")
        original_rotation = vector(client, f"vget {camera_path}/rotation", 3)
        original_fov = vector(client, f"vget {camera_path}/fov", 1)[0]
        original_size = vector(client, f"vget {camera_path}/size", 2)
        result["location"] = vector(client, f"vget {camera_path}/location", 3)
        result["original_rotation"] = original_rotation
        result["original_fov"] = original_fov
        result["original_size"] = original_size
        text_response(client, f"vset {camera_path}/size 960 540")
        time.sleep(args.settle_seconds)
        before = take("before")
        time.sleep(args.settle_seconds)
        repeat = take("before_repeat")
        noise = lower_half_difference(before, repeat)
        threshold = max(2.0, noise * 5)
        result["image_effects"].update({"baseline_difference": noise,
                                         "required_difference": threshold,
                                         "region": "lower half RGB; mean absolute difference (0-255)"})

        pitch, yaw, roll = original_rotation
        text_response(client, f"vset {camera_path}/rotation {pitch} {yaw + 30} {roll}")
        time.sleep(args.settle_seconds)
        rotation = vector(client, f"vget {camera_path}/rotation", 3)
        result["checks"]["pan"] = all(close_angle(a, b) for a, b in zip(rotation, [pitch, yaw + 30, roll]))
        result["image_effects"]["pan_difference"] = lower_half_difference(before, take("pan"))
        result["checks"]["pan_image_changed"] = result["image_effects"]["pan_difference"] > threshold

        new_pitch = pitch - 20 if pitch >= -60 else pitch + 20
        text_response(client, f"vset {camera_path}/rotation {new_pitch} {yaw} {roll}")
        time.sleep(args.settle_seconds)
        rotation = vector(client, f"vget {camera_path}/rotation", 3)
        result["checks"]["tilt"] = all(close_angle(a, b) for a, b in zip(rotation, [new_pitch, yaw, roll]))
        result["image_effects"]["tilt_difference"] = lower_half_difference(before, take("tilt"))
        result["checks"]["tilt_image_changed"] = result["image_effects"]["tilt_difference"] > threshold

        text_response(client, f"vset {camera_path}/rotation {pitch} {yaw} {roll}")
        zoom_fov = original_fov * 0.7
        text_response(client, f"vset {camera_path}/fov {zoom_fov}")
        time.sleep(args.settle_seconds)
        result["checks"]["zoom"] = abs(vector(client, f"vget {camera_path}/fov", 1)[0] - zoom_fov) < 0.1
        result["image_effects"]["zoom_difference"] = lower_half_difference(before, take("zoom"))
        result["checks"]["zoom_image_changed"] = result["image_effects"]["zoom_difference"] > threshold
        result["checks"]["image_resolution"] = all(
            item["width"] == 960 and item["height"] == 540 for item in result["captures"]
        )
    finally:
        try:
            if client.isconnected():
                if original_rotation is not None:
                    text_response(client, f"vset {camera_path}/rotation " + " ".join(str(value) for value in original_rotation))
                if original_fov is not None:
                    text_response(client, f"vset {camera_path}/fov {original_fov}")
                if original_size is not None:
                    text_response(client, f"vset {camera_path}/size {int(original_size[0])} {int(original_size[1])}")
                if original_rotation is not None and original_fov is not None and original_size is not None:
                    time.sleep(args.settle_seconds)
                    rotation = vector(client, f"vget {camera_path}/rotation", 3)
                    fov = vector(client, f"vget {camera_path}/fov", 1)[0]
                    size = vector(client, f"vget {camera_path}/size", 2)
                    result["checks"]["restored"] = all(
                        close_angle(a, b) for a, b in zip(rotation, original_rotation)
                    ) and abs(fov - original_fov) < 0.1 and size == original_size
        except Exception as error:
            result["checks"]["restored"] = False
            result["restore_error"] = str(error)
        finally:
            client.disconnect()
        expected = {"pan", "tilt", "zoom", "pan_image_changed", "tilt_image_changed",
                    "zoom_image_changed", "image_resolution", "restored"}
        result["ok"] = expected.issubset(result["checks"]) and all(result["checks"].values())
        (args.output / "unrealcv-verification.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit("One or more UnrealCV checks failed")


if __name__ == "__main__":
    main()
