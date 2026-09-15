from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from tqdm import tqdm

from src.progress import ProgressTracker
from src.topology import get_edges, get_joint_names

try:
    import smplx
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "smplx is required. Install dependencies with `pip install -r requirements.txt`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 3D joints from AMASS parameter files."
    )
    parser.add_argument("--amass-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-type", choices=["smplx", "smplh"], default="smplx")
    parser.add_argument("--ext", default=".npz")
    parser.add_argument("--num-joints", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--root-index", type=int, default=0)
    parser.add_argument("--save-root-relative", action="store_true")
    parser.add_argument("--include-subsets", nargs="*", default=None)
    parser.add_argument("--progress-file", type=Path, default=Path("runs/progress/extract_progress.json"))
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def to_tensor(array: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def normalize_gender(gender_value) -> str:
    if isinstance(gender_value, np.ndarray):
        if gender_value.ndim == 0:
            gender_value = gender_value.item()
        else:
            gender_value = gender_value[0]
    if isinstance(gender_value, bytes):
        gender_value = gender_value.decode("utf-8")
    value = str(gender_value).strip().lower()
    if "female" in value:
        return "female"
    if "male" in value:
        return "male"
    return "neutral"


def split_pose_components(data: Dict[str, np.ndarray], model_type: str) -> Dict[str, np.ndarray]:
    if "poses" in data:
        poses = np.asarray(data["poses"], dtype=np.float32)
        if model_type == "smplh":
            return {
                "global_orient": poses[:, 0:3],
                "body_pose": poses[:, 3:66],
                "left_hand_pose": poses[:, 66:111],
                "right_hand_pose": poses[:, 111:156],
            }

        components = {
            "global_orient": poses[:, 0:3],
            "body_pose": poses[:, 3:66],
            "jaw_pose": np.zeros((poses.shape[0], 3), dtype=np.float32),
            "leye_pose": np.zeros((poses.shape[0], 3), dtype=np.float32),
            "reye_pose": np.zeros((poses.shape[0], 3), dtype=np.float32),
            "left_hand_pose": np.zeros((poses.shape[0], 45), dtype=np.float32),
            "right_hand_pose": np.zeros((poses.shape[0], 45), dtype=np.float32),
        }
        if poses.shape[1] >= 111:
            components["left_hand_pose"] = poses[:, 66:111]
        if poses.shape[1] >= 156:
            components["right_hand_pose"] = poses[:, 111:156]
        if poses.shape[1] >= 165:
            components["jaw_pose"] = poses[:, 66:69]
            components["leye_pose"] = poses[:, 69:72]
            components["reye_pose"] = poses[:, 72:75]
            components["left_hand_pose"] = poses[:, 75:120]
            components["right_hand_pose"] = poses[:, 120:165]
        return components

    components = {
        "global_orient": np.asarray(
            data.get("global_orient", data.get("root_orient")), dtype=np.float32
        ),
        "body_pose": np.asarray(data["body_pose"], dtype=np.float32),
    }
    optional = [
        "left_hand_pose",
        "right_hand_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
    ]
    for key in optional:
        value = data.get(key)
        if value is not None:
            components[key] = np.asarray(value, dtype=np.float32)
    return components


def build_model(model_path: Path, model_type: str, gender: str, batch_size: int):
    model_gender = {"female": "female", "male": "male"}.get(gender, "neutral")
    resolved_model_path = model_path
    nested_model_path = model_path / model_type
    if nested_model_path.exists():
        resolved_model_path = model_path
    elif any((model_path / name).exists() for name in (
        f"{model_type.upper()}_NEUTRAL.npz",
        f"{model_type.upper()}_MALE.npz",
        f"{model_type.upper()}_FEMALE.npz",
    )):
        resolved_model_path = model_path.parent
    return smplx.create(
        model_path=str(resolved_model_path),
        model_type=model_type,
        gender=model_gender,
        use_pca=False,
        batch_size=batch_size,
    )


def iter_npz_files(root: Path, ext: str) -> Iterable[Path]:
    return sorted(path for path in root.rglob(f"*{ext}") if path.is_file())


def filter_subset_files(files: Iterable[Path], root: Path, include_subsets: list[str] | None) -> list[Path]:
    if not include_subsets:
        return list(files)
    allowed = {item.lower() for item in include_subsets}
    selected = []
    for path in files:
        try:
            first_part = path.relative_to(root).parts[0].lower()
        except Exception:
            continue
        if first_part in allowed:
            selected.append(path)
    return selected


def default_manifest_path(output_root: Path, include_subsets: list[str] | None) -> Path:
    if include_subsets:
        subset_tag = "_".join(item.lower() for item in include_subsets)
        return output_root / f"manifest_{subset_tag}.json"
    return output_root / "manifest.json"


def looks_like_amass_sequence(data: Dict[str, np.ndarray]) -> bool:
    if "poses" in data:
        return True
    if "body_pose" in data:
        return True
    return False


def extract_sequence(
    file_path: Path,
    amass_root: Path,
    output_root: Path,
    model_path: Path,
    model_type: str,
    num_joints: int,
    batch_size: int,
    root_index: int,
    save_root_relative: bool,
    device: str,
) -> Dict[str, object]:
    data = np.load(file_path, allow_pickle=True)
    if not looks_like_amass_sequence(data):
        raise ValueError("Not an AMASS motion parameter file")
    components = split_pose_components(data, model_type)
    trans = np.asarray(data.get("trans", np.zeros((components["body_pose"].shape[0], 3))), dtype=np.float32)
    betas = np.asarray(data.get("betas", np.zeros(16, dtype=np.float32)), dtype=np.float32)
    num_frames = int(components["body_pose"].shape[0])
    gender = normalize_gender(data.get("gender", "neutral"))
    fps_value = data.get("mocap_framerate", data.get("mocap_frame_rate", 30.0))
    mocap_fps = float(np.asarray(fps_value).reshape(-1)[0])

    model = build_model(model_path, model_type, gender, batch_size=batch_size).to(device)
    joints_batches = []
    betas_batch = np.repeat(betas[None, :10], repeats=1, axis=0)

    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        current_batch = end - start
        current = {}
        for key, value in components.items():
            batch_value = value[start:end]
            if current_batch < batch_size:
                pad_width = ((0, batch_size - current_batch), (0, 0))
                batch_value = np.pad(batch_value, pad_width, mode="constant")
            current[key] = to_tensor(batch_value, device)
        transl_value = trans[start:end]
        if current_batch < batch_size:
            transl_value = np.pad(transl_value, ((0, batch_size - current_batch), (0, 0)), mode="constant")
        current["transl"] = to_tensor(transl_value, device)
        current["betas"] = to_tensor(np.repeat(betas_batch, batch_size, axis=0), device)

        with torch.no_grad():
            output = model(**current)
            joints = (
                output.joints[:current_batch, :num_joints, :]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        joints_batches.append(joints)

    joints = np.concatenate(joints_batches, axis=0)
    root_positions = joints[:, root_index : root_index + 1, :].copy()
    joints_root_relative = joints - root_positions
    relative_path = file_path.relative_to(amass_root)
    target_path = output_root / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    save_payload = {
        "joints": joints,
        "root_positions": root_positions.astype(np.float32),
        "trans": trans,
        "fps": mocap_fps,
        "gender": gender,
        "source": str(relative_path).replace("\\", "/"),
        "joint_names": np.asarray(get_joint_names(num_joints), dtype=object),
        "skeleton_edges": np.asarray(get_edges(num_joints), dtype=np.int64),
        "root_index": root_index,
    }
    if save_root_relative:
        save_payload["joints_root_relative"] = joints_root_relative.astype(np.float32)
    np.savez_compressed(target_path, **save_payload)

    return {
        "source": str(relative_path).replace("\\", "/"),
        "frames": num_frames,
        "fps": mocap_fps,
        "gender": gender,
        "root_index": root_index,
        "output": str(target_path.relative_to(output_root)).replace("\\", "/"),
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    files = [path for path in iter_npz_files(args.amass_root, args.ext) if "stageii" in path.name.lower()]
    files = filter_subset_files(files, args.amass_root, args.include_subsets)
    if not files:
        raise SystemExit(f"No files found under {args.amass_root}")

    manifest = []
    tracker = ProgressTracker(args.progress_file)
    tracker.start(
        stage="extract_joints",
        total=len(files),
        extra={
            "amass_root": str(args.amass_root),
            "output_root": str(args.output_root),
            "model_type": args.model_type,
            "include_subsets": args.include_subsets or [],
        },
    )
    success_count = 0
    failed_count = 0
    progress_bar = tqdm(files, desc="Extracting joints", unit="seq")
    for index, file_path in enumerate(progress_bar, start=1):
        source = str(file_path.relative_to(args.amass_root)).replace("\\", "/")
        try:
            info = extract_sequence(
                file_path=file_path,
                amass_root=args.amass_root,
                output_root=args.output_root,
                model_path=args.model_path,
                model_type=args.model_type,
                num_joints=args.num_joints,
                batch_size=args.batch_size,
                root_index=args.root_index,
                save_root_relative=args.save_root_relative,
                device=args.device,
            )
            manifest.append(info)
            success_count += 1
            tracker.update(
                processed=index,
                success=success_count,
                failed=failed_count,
                current_item=source,
            )
            progress_bar.set_postfix(tracker.tqdm_postfix())
        except Exception as exc:  # pragma: no cover
            failed_count += 1
            manifest.append(
                {
                    "source": source,
                    "error": str(exc),
                }
            )
            tracker.update(
                processed=index,
                success=success_count,
                failed=failed_count,
                current_item=source,
                last_error=str(exc),
            )
            progress_bar.set_postfix(tracker.tqdm_postfix())

    summary_path = args.manifest_path or default_manifest_path(args.output_root, args.include_subsets)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "num_sequences": len(files),
                "model_type": args.model_type,
                "num_joints": args.num_joints,
                "root_index": args.root_index,
                "save_root_relative": args.save_root_relative,
                "include_subsets": args.include_subsets or [],
                "joint_names": get_joint_names(args.num_joints),
                "skeleton_edges": get_edges(args.num_joints),
                "items": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    tracker.finish(
        status="completed",
        extra={
            "manifest_path": str(summary_path),
            "num_sequences": len(files),
            "success": success_count,
            "failed": failed_count,
        },
    )
    print(f"Saved manifest to {summary_path}")


if __name__ == "__main__":
    main()
