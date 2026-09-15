from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.progress import ProgressTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze AMASS subset metadata.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--ext", default=".npz")
    parser.add_argument("--output-json", type=Path, default=Path("runs/reports/amass_analysis.json"))
    parser.add_argument("--output-md", type=Path, default=Path("runs/reports/amass_analysis.md"))
    parser.add_argument("--progress-file", type=Path, default=Path("runs/progress/amass_analysis_progress.json"))
    return parser.parse_args()


def sequence_files(root: Path, ext: str):
    return sorted(path for path in root.rglob(f"*{ext}") if path.is_file())


def safe_shape(value):
    return tuple(int(x) for x in getattr(value, "shape", ()))


def main() -> None:
    args = parse_args()
    files = sequence_files(args.input_root, args.ext)
    if not files:
        raise SystemExit(f"No files found under {args.input_root}")

    subject_counts = Counter()
    key_counts = Counter()
    gender_counts = Counter()
    frame_counts = []
    pose_dims = Counter()
    fps_counts = Counter()
    sample_summaries = []
    unreadable_files = []
    readable_count = 0
    tracker = ProgressTracker(args.progress_file)
    tracker.start(
        stage="analyze_amass",
        total=len(files),
        extra={"input_root": str(args.input_root), "output_json": str(args.output_json)},
    )

    progress_bar = tqdm(files, desc="Analyzing dataset", unit="file")
    for index, file_path in enumerate(progress_bar, start=1):
        relative = file_path.relative_to(args.input_root).as_posix()
        parts = relative.split("/")
        subject = parts[0] if parts else relative
        subject_counts[subject] += 1

        try:
            with np.load(file_path, allow_pickle=True) as data:
                readable_count += 1
                key_counts.update(list(data.keys()))
                if "gender" in data:
                    gender = str(np.asarray(data["gender"]).item())
                    gender_counts[gender.lower()] += 1

                if "poses" in data:
                    frames = int(data["poses"].shape[0])
                    pose_dims[str(int(data["poses"].shape[1]))] += 1
                elif "body_pose" in data:
                    frames = int(data["body_pose"].shape[0])
                    pose_dims[str(int(data["body_pose"].shape[1]))] += 1
                else:
                    frames = 0
                if frames:
                    frame_counts.append(frames)
                fps_value = data.get("mocap_framerate", data.get("mocap_frame_rate", None))
                if fps_value is not None:
                    fps = float(np.asarray(fps_value).reshape(-1)[0])
                    fps_counts[str(fps)] += 1

                if len(sample_summaries) < 10:
                    sample_summaries.append(
                        {
                            "file": relative,
                            "keys": sorted(list(data.keys())),
                            "frames": frames,
                            "pose_shape": safe_shape(data.get("poses", data.get("body_pose", np.empty((0, 0))))),
                            "trans_shape": safe_shape(data.get("trans", np.empty((0, 0)))),
                        }
                    )
            tracker.update(
                processed=index,
                success=readable_count,
                failed=len(unreadable_files),
                current_item=relative,
            )
            progress_bar.set_postfix(tracker.tqdm_postfix())
        except Exception as exc:
            unreadable_files.append({"file": relative, "error": str(exc)})
            tracker.update(
                processed=index,
                success=readable_count,
                failed=len(unreadable_files),
                current_item=relative,
                last_error=str(exc),
            )
            progress_bar.set_postfix(tracker.tqdm_postfix())

    report = {
        "input_root": str(args.input_root),
        "num_files": len(files),
        "readable_files": readable_count,
        "unreadable_files": unreadable_files,
        "subjects": len(subject_counts),
        "subject_counts": dict(subject_counts),
        "key_counts": dict(key_counts),
        "gender_counts": dict(gender_counts),
        "frames": {
            "min": min(frame_counts) if frame_counts else None,
            "max": max(frame_counts) if frame_counts else None,
            "mean": float(sum(frame_counts) / len(frame_counts)) if frame_counts else None,
        },
        "pose_dims": dict(pose_dims),
        "fps_counts": dict(fps_counts),
        "sample_summaries": sample_summaries,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# AMASS analysis report: {args.input_root.name}",
        "",
        f"- Input root: `{args.input_root}`",
        f"- Num files: `{len(files)}`",
        f"- Readable files: `{readable_count}`",
        f"- Unreadable files: `{len(unreadable_files)}`",
        f"- Num subjects: `{len(subject_counts)}`",
        f"- Frames min: `{report['frames']['min']}`",
        f"- Frames max: `{report['frames']['max']}`",
        f"- Frames mean: `{report['frames']['mean']}`",
        "",
        "## Most common subjects",
    ]
    for subject, count in subject_counts.most_common(10):
        lines.append(f"- {subject}: {count}")
    lines.append("")
    lines.append("## Common keys")
    for key, count in key_counts.most_common(15):
        lines.append(f"- {key}: {count}")
    lines.append("")
    if unreadable_files:
        lines.append("## Unreadable files")
        for item in unreadable_files[:20]:
            lines.append(f"- {item['file']}: {item['error']}")
        lines.append("")
    lines.append("## Sample files")
    for item in sample_summaries:
        lines.append(f"- {item['file']}: frames={item['frames']}, pose_shape={item['pose_shape']}, trans_shape={item['trans_shape']}")

    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    tracker.finish(
        status="completed",
        extra={
            "readable_files": readable_count,
            "unreadable_files": len(unreadable_files),
            "output_json": str(args.output_json),
            "output_md": str(args.output_md),
        },
    )
    print(args.output_md)
    print(args.output_json)


if __name__ == "__main__":
    main()
