"""Report every completed training round without changing the running learner."""
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


REPRO = Path(__file__).resolve().parents[1]
DEFAULT_RUN = REPRO / "runs/gated-static-center-v1"
FIELDS = ("round", "seed", "q_loss_mean", "parameter_objective_mean", "parameter_grad_norm_mean",
          "reward_sum", "center_mean", "center_p95", "direction_reversals", "epsilon", "steps",
          "network_updates", "updates_total", "passed", "recent_32_pass_rate", "q_learning_rate",
          "parameter_learning_rate", "batch_size", "wall_seconds", "update_log_missing", "nonfinite_fields")


def read_complete_jsonl(path):
    """A currently written final line is ignored until its newline arrives."""
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8-sig")
    if text and not text.endswith("\n"):
        text = text.rsplit("\n", 1)[0] if "\n" in text else ""
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid completed JSONL row: {path}:{number}") from error
    return rows


def update_mean(rows, key, nonfinite):
    values = [row[key] for row in rows if row.get(key) is not None]
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        nonfinite.append(key)
        return None
    return statistics.fmean(values) if values else None


def collect(run, cache=None):
    run = Path(run)
    config = json.loads((run / "configuration.json").read_text(encoding="utf-8-sig"))
    episodes = read_complete_jsonl(run / "episodes.jsonl")
    cache = cache if cache is not None else {}
    output = []
    for index, episode in enumerate(episodes):
        key = (index, episode["seed"])
        if key not in cache:
            update_path = run / "training" / f"episode-{index:04d}-seed{episode['seed']}" / "updates.jsonl"
            updates = read_complete_jsonl(update_path)
            nonfinite = []
            q_loss = update_mean(updates, "q_loss", nonfinite)
            if not updates:
                q_loss = episode.get("mean_q_loss")
            cache[key] = {"round": index + 1, "seed": episode["seed"],
                "q_loss_mean": q_loss,
                "parameter_objective_mean": update_mean(updates, "parameter_objective", nonfinite),
                "parameter_grad_norm_mean": update_mean(updates, "parameter_grad_norm", nonfinite),
                "reward_sum": episode.get("reward_sum"), "center_mean": episode.get("terminal_center_mean"),
                "center_p95": episode.get("terminal_center_p95"),
                "direction_reversals": episode.get("direction_reversals"), "epsilon": episode.get("epsilon"),
                "steps": episode.get("steps"), "network_updates": episode.get("network_updates"),
                "updates_total": episode.get("updates_total"), "passed": episode.get("passed") is True,
                "q_learning_rate": config["pdqn"]["actor_lr"],
                "parameter_learning_rate": config["pdqn"]["param_lr"],
                "batch_size": config["pdqn"]["batch_size"], "wall_seconds": episode.get("wall_seconds"),
                "update_log_missing": not update_path.exists(), "nonfinite_fields": nonfinite}
        row = dict(cache[key])
        recent = episodes[max(0, index - 31):index + 1]
        row["recent_32_pass_rate"] = sum(item.get("passed") is True for item in recent) / len(recent)
        output.append(row)
    return output


def _write_snapshot(path, contents, encoding):
    """A viewer may permit writes but deny Windows replace/delete sharing."""
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(contents, encoding=encoding, newline="")
        for attempt in range(3):
            try:
                temporary.replace(path)
                return None
            except PermissionError:
                if attempt < 2:
                    time.sleep(.05 * (attempt + 1))
        # Human viewers commonly hold a read handle without FILE_SHARE_DELETE.
        # A direct update is sufficient for these non-scientific sidecars.
        path.write_text(contents, encoding=encoding, newline="")
        return None
    except OSError as error:
        # Broadcasting must continue even if a viewer also denies write access.
        return f"Sidecar temporarily unavailable: {path.name}: {error}"
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def save_sidecars(output, rows):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    json_text = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                         for key, value in row.items()})
    header = (f"Q学习率={rows[0]['q_learning_rate']} | 时长学习率={rows[0]['parameter_learning_rate']} | "
              f"batch={rows[0]['batch_size']}\n") if rows else "暂无已完成回合\n"
    human_text = header + "时长目标为最大化Q目标；本回合门槛不等于阶段40例验证通过。\n"
    human_text += "".join(formatted(row) + "\n" for row in rows)
    warnings = []
    for name, contents, encoding in (("round-metrics.jsonl", json_text, "utf-8"),
                                     ("round-metrics.csv", buffer.getvalue(), "utf-8-sig"),
                                     ("round-metrics.log", human_text, "utf-8")):
        path = output / name
        warning = _write_snapshot(path, contents, encoding)
        if warning:
            warnings.append(warning)
    return warnings


def formatted(row):
    value = lambda key, digits=4: "未更新/缺失" if row.get(key) is None else f"{row[key]:.{digits}f}"
    return (f"第{row['round']}轮 | Q损失={value('q_loss_mean', 5)} | "
            f"时长目标(最大化)={value('parameter_objective_mean', 3)} | "
            f"梯度范数={value('parameter_grad_norm_mean', 3)} | 奖励={value('reward_sum', 3)} | "
            f"中心均值/P95={value('center_mean')}/{value('center_p95')} | "
            f"反向={row['direction_reversals']} | ε={value('epsilon', 3)} | "
            f"步数={row['steps']} | 累计更新={row['updates_total']} | "
            f"{'本回合过门槛' if row['passed'] else '本回合未过门槛'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="single snapshot (default)")
    mode.add_argument("--follow", action="store_true", help="print each new completed round immediately")
    parser.add_argument("--last", type=int, help="print only the last N existing rounds, then new rounds")
    parser.add_argument("--after", type=int, default=0, help="print rounds with a number greater than N")
    parser.add_argument("--json", action="store_true", help="print machine-readable metrics instead of human lines")
    parser.add_argument("--no-save", action="store_true", help="read/print only; avoid sidecar writes while another watcher follows")
    parser.add_argument("--poll-seconds", type=float, default=1.)
    parser.add_argument("--timeout", type=float, default=0., help="bounded follow duration; zero waits for run completion")
    args = parser.parse_args(argv)
    if args.after < 0 or (args.last is not None and args.last < 1) or args.poll_seconds < 1 or args.timeout < 0:
        parser.error("after>=0, last>=1, poll-seconds>=1 and timeout>=0 are required")
    run = args.run.resolve()
    if not (run / "configuration.json").is_file():
        parser.error(f"Training configuration missing: {run}")
    output = (args.output or REPRO / "monitoring" / run.name).resolve()
    if output == run or run in output.parents:
        parser.error("Reporter output must be outside the scientific run directory")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cache, printed, first, warned = {}, args.after, True, set()
    started, stopped_since = time.monotonic(), None
    while True:
        rows = collect(run, cache)
        if not args.no_save:
            for warning in save_sidecars(output, rows):
                key = warning.split(":", 2)[1]
                if key not in warned:
                    print(warning, file=sys.stderr, flush=True)
                    warned.add(key)
        threshold = max(printed, len(rows) - args.last) if first and args.last is not None else printed
        for row in rows:
            if row["round"] > threshold:
                print(json.dumps(row, ensure_ascii=False, allow_nan=False) if args.json else formatted(row), flush=True)
        printed = max(printed, len(rows))
        first = False
        if not args.follow:
            break
        if (run / "results.json").is_file():
            break
        launcher = run / "launcher.json"
        if launcher.exists():
            info = json.loads(launcher.read_text(encoding="utf-8-sig"))
            if info.get("owned_process_stopped"):
                stopped_since = stopped_since or time.monotonic()
        if stopped_since is not None and time.monotonic() - stopped_since >= 3.:
            break
        if args.timeout and time.monotonic() - started >= args.timeout:
            break
        time.sleep(args.poll_seconds)
    return rows


if __name__ == "__main__":
    main()
