from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_hisrep_cmu_clean as hisrep_train  # noqa: E402
import train_ltc_topology_rollout_consistency as ltc_train  # noqa: E402


DATASETS = ("cmu", "kit", "bmlmovi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runtime smoke test for Protocol v2 model paths.")
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mirror-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--optimizer-steps", type=int, default=2)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def finite_gradients(model: nn.Module) -> tuple[bool, float]:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        return False, 0.0
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    squared_norm = sum(torch.sum(gradient.detach().double() ** 2) for gradient in gradients)
    norm = float(torch.sqrt(squared_norm).cpu())
    return finite and np.isfinite(norm) and norm > 0.0, norm


class RolloutTrace(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.inputs: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.inputs.append(inputs.detach().clone())
        outputs = self.model(inputs)
        self.outputs.append(outputs.detach().clone())
        return outputs


def protocol_args() -> SimpleNamespace:
    return SimpleNamespace(
        history=25,
        future_step=25,
        horizon=75,
        stride=5,
        expected_manifest_sha256="",
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        lambda_26_50=0.7,
        lambda_51_75=0.5,
        lambda_bone=0.1,
        lambda_root=0.1,
        lambda_velocity=0.05,
    )


def smoke_ltc(
    dataset_slug: str,
    representation: str,
    protocol_root: Path,
    manifest_hash: str,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    optimizer_steps: int,
) -> dict:
    data_root = protocol_root / dataset_slug / representation
    metadata_path = data_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    args = protocol_args()
    args.expected_manifest_sha256 = manifest_hash
    ltc_train.validate_protocol_v2_metadata(metadata, args)
    dataset = ltc_train.RolloutWindowDataset(
        dataset_name=metadata["dataset"],
        metadata_path=metadata_path,
        split="train",
        history=25,
        horizon=75,
        stride=5,
        stats_file=data_root / "normalization_stats.npz",
        max_windows=max(batch_size * optimizer_steps, batch_size),
        seed=42,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    edges_list = [[int(src), int(dst)] for src, dst in metadata["skeleton_edges"]]
    guidance_index = int(metadata["guidance_joint_index"])
    model = ltc_train.make_model(
        args,
        joints=int(metadata["num_joints"]),
        history=25,
        future=25,
        skeleton_edges=edges_list,
        guidance_joint_index=guidance_index,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
    edges = torch.tensor(edges_list, dtype=torch.long, device=device)
    losses: list[float] = []
    root_feature_grad_norms: list[float] = []
    rollout_checks: dict[str, object] = {}
    last_inputs: torch.Tensor | None = None

    model.train()
    for step_index, batch in enumerate(loader):
        if step_index >= optimizer_steps:
            break
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        traced = RolloutTrace(model)
        predictions = ltc_train.recursive_rollout(traced, inputs, horizon=75, step=25)
        if step_index == 0:
            rollout_checks = {
                "calls": len(traced.inputs),
                "input_shapes": [list(tensor.shape) for tensor in traced.inputs],
                "output_shape": list(predictions.shape),
                "stage2_equals_stage1_prediction": bool(torch.equal(traced.inputs[1], traced.outputs[0])),
                "stage3_equals_stage2_prediction": bool(torch.equal(traced.inputs[2], traced.outputs[1])),
            }
        loss, _ = ltc_train.rollout_loss(
            inputs,
            predictions,
            targets,
            edges,
            args.lambda_26_50,
            args.lambda_51_75,
            args.lambda_bone,
            args.lambda_root,
            args.lambda_velocity,
            guidance_index,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        temporal_weight = model.encoder.temporal_encoder.layers[0].input_linear.weight
        node_hidden = model.encoder.node_hidden
        root_feature_grad = temporal_weight.grad[:, node_hidden:]
        root_feature_grad_norms.append(float(torch.linalg.vector_norm(root_feature_grad).detach().cpu()))
        gradients_ok, gradient_norm = finite_gradients(model)
        if not gradients_ok:
            raise RuntimeError(f"Invalid LTC gradients for {dataset_slug}/{representation}: {gradient_norm}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        last_inputs = inputs.detach()

    if last_inputs is None or len(losses) != optimizer_steps:
        raise RuntimeError(f"Insufficient LTC smoke batches for {dataset_slug}/{representation}")
    model.eval()
    checkpoint_path = output_dir / f"checkpoint_ltc_{dataset_slug}_{representation}.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint_path)
    clone = ltc_train.make_model(
        args,
        joints=int(metadata["num_joints"]),
        history=25,
        future=25,
        skeleton_edges=edges_list,
        guidance_joint_index=guidance_index,
    ).to(device)
    clone.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state"])
    clone.eval()
    with torch.no_grad():
        original_output = model(last_inputs)
        reloaded_output = clone(last_inputs)
    reload_diff = float(torch.max(torch.abs(original_output - reloaded_output)).cpu())
    checkpoint_path.unlink()
    all_pass = (
        all(np.isfinite(losses))
        and min(root_feature_grad_norms) > 0.0
        and rollout_checks["calls"] == 3
        and all(shape[1] == 25 for shape in rollout_checks["input_shapes"])
        and rollout_checks["stage2_equals_stage1_prediction"]
        and rollout_checks["stage3_equals_stage2_prediction"]
        and reload_diff == 0.0
    )
    return {
        "dataset": metadata["dataset"],
        "representation": representation,
        "model": "Full LTC-Topology",
        "joints": int(metadata["num_joints"]),
        "guidance_joint_index": guidance_index,
        "guidance_joint_name": metadata["guidance_joint_name"],
        "skeleton_edges": len(edges_list),
        "losses": losses,
        "root_feature_gradient_norms": root_feature_grad_norms,
        "rollout": rollout_checks,
        "checkpoint_reload_max_diff": reload_diff,
        "all_checks_pass": bool(all_pass),
    }


def hisrep_predict_block(model: nn.Module, current_m: torch.Tensor, cfg: hisrep_train.Config) -> torch.Tensor:
    batch_size = current_m.shape[0]
    padded_m = torch.cat([current_m, torch.zeros_like(current_m)], dim=1)
    src_mm = (padded_m * hisrep_train.MM_PER_METER).reshape(
        batch_size, cfg.input_n + cfg.output_n, cfg.in_features
    )
    out_all = model(src_mm, output_n=cfg.model_output_n, input_n=cfg.input_n, itera=cfg.itera)
    return hisrep_train._hisrep_future_from_output(out_all, cfg, batch_size) / hisrep_train.MM_PER_METER


def smoke_hisrep(
    dataset_slug: str,
    protocol_root: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    optimizer_steps: int,
) -> dict:
    data_root = protocol_root / dataset_slug / "common18"
    metadata = json.loads((data_root / "metadata.json").read_text(encoding="utf-8"))
    inputs = np.load(data_root / "train_inputs.npy", mmap_mode="r")
    targets = np.load(data_root / "train_targets.npy", mmap_mode="r")
    cfg = hisrep_train.Config(
        dataset=metadata["dataset"],
        input_n=25,
        output_n=25,
        model_output_n=10,
        itera=3,
        dct_n=20,
        batch_size=batch_size,
        test_batch_size=batch_size,
        epochs=1,
        num_workers=0,
        disable_tqdm=True,
    )
    model = hisrep_train.make_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    losses: list[float] = []
    gradient_norms: list[float] = []
    last_history: torch.Tensor | None = None
    model.train()
    for step_index in range(optimizer_steps):
        start = step_index * batch_size
        stop = start + batch_size
        batch_m = torch.from_numpy(
            np.concatenate([np.asarray(inputs[start:stop]), np.asarray(targets[start:stop])], axis=1).copy()
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = hisrep_train.forward_hisrep(model, batch_m, cfg)
        loss.backward()
        gradients_ok, gradient_norm = finite_gradients(model)
        if not gradients_ok:
            raise RuntimeError(f"Invalid HisRep gradients for {dataset_slug}: {gradient_norm}")
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(gradient_norm)
        last_history = batch_m[:, :25].detach()

    if last_history is None:
        raise RuntimeError(f"No HisRep smoke batch for {dataset_slug}")
    model.eval()
    rollout_inputs: list[torch.Tensor] = []
    rollout_outputs: list[torch.Tensor] = []
    current = last_history
    with torch.no_grad():
        for _ in range(3):
            rollout_inputs.append(current.detach().clone())
            predicted = hisrep_predict_block(model, current, cfg)
            rollout_outputs.append(predicted.detach().clone())
            current = predicted
    rollout = torch.cat(rollout_outputs, dim=1)
    checkpoint_path = output_dir / f"checkpoint_hisrep_{dataset_slug}.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint_path)
    clone = hisrep_train.make_model(cfg, device)
    clone.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state"])
    clone.eval()
    with torch.no_grad():
        original_output = hisrep_predict_block(model, last_history, cfg)
        reloaded_output = hisrep_predict_block(clone, last_history, cfg)
    reload_diff = float(torch.max(torch.abs(original_output - reloaded_output)).cpu())
    checkpoint_path.unlink()
    input_shapes = [list(tensor.shape) for tensor in rollout_inputs]
    all_pass = (
        all(np.isfinite(losses))
        and min(gradient_norms) > 0.0
        and list(rollout.shape)[1:] == [75, 18, 3]
        and all(shape[1:] == [25, 18, 3] for shape in input_shapes)
        and bool(torch.equal(rollout_inputs[1], rollout_outputs[0]))
        and bool(torch.equal(rollout_inputs[2], rollout_outputs[1]))
        and reload_diff == 0.0
    )
    return {
        "dataset": metadata["dataset"],
        "representation": "common18",
        "model": "HisRepItself",
        "joints": 18,
        "losses_mm": losses,
        "gradient_norms": gradient_norms,
        "rollout": {
            "calls": 3,
            "input_shapes": input_shapes,
            "output_shape": list(rollout.shape),
            "stage2_equals_stage1_prediction": bool(torch.equal(rollout_inputs[1], rollout_outputs[0])),
            "stage3_equals_stage2_prediction": bool(torch.equal(rollout_inputs[2], rollout_outputs[1])),
        },
        "checkpoint_reload_max_diff_m": reload_diff,
        "all_checks_pass": bool(all_pass),
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the HisRepItself runtime smoke path")
    device = torch.device(args.device)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash = sha256_file(args.frozen_manifest)
    expected_hash = "2e4ed5afb524c328efdd1246fa61fb96e104b39484e17d55784fa3ef2aec879e"
    if manifest_hash.lower() != expected_hash:
        raise RuntimeError(f"Unexpected frozen manifest hash: {manifest_hash}")

    results: list[dict] = []
    for dataset_slug in DATASETS:
        for representation in ("native25", "common18"):
            print(f"SMOKE LTC {dataset_slug} {representation}", flush=True)
            results.append(
                smoke_ltc(
                    dataset_slug,
                    representation,
                    args.protocol_root,
                    manifest_hash,
                    args.output_dir,
                    device,
                    args.batch_size,
                    args.optimizer_steps,
                )
            )
        print(f"SMOKE HisRepItself {dataset_slug} common18", flush=True)
        results.append(
            smoke_hisrep(
                dataset_slug,
                args.protocol_root,
                args.output_dir,
                device,
                args.batch_size,
                args.optimizer_steps,
            )
        )

    failed = [result for result in results if not result["all_checks_pass"]]
    report = {
        "status": "PASS" if not failed else "FAIL",
        "formal_training_started": False,
        "optimizer_smoke_steps_per_path": args.optimizer_steps,
        "device": str(device),
        "frozen_manifest": str(args.frozen_manifest),
        "frozen_manifest_sha256": manifest_hash,
        "reviewer_scope": {
            "R1_comment_1_split_and_leakage": "covered by frozen subject manifest and data smoke report",
            "R1_comment_2_recursive_rollout": "runtime-traced with exactly three fixed-length latest-25 inputs",
            "R1_comment_3_root_guidance": "runtime-tested as graph-encoded anchor feature plus anchor-coordinate loss",
            "R1_comment_5_preprocessing_control": "common18 physical arrays are shared before model-native scaling",
            "outside_this_smoke": ["modern baselines", "matched multi-seed training", "loss sensitivity", "formal accuracy and timing"],
        },
        "results": results,
        "failed_paths": [f"{item['dataset']}:{item['model']}:{item['representation']}" for item in failed],
    }
    json_path = args.output_dir / "protocol_v2_runtime_smoke_test.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Protocol v2 Runtime Smoke Test",
        "",
        f"Status: **{report['status']}**",
        "",
        "No formal training was started; each path used two optimizer smoke steps.",
        "",
        "| Dataset | Model | Representation | Joints | Pass |",
        "|---|---|---|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['dataset']} | {result['model']} | {result['representation']} | "
            f"{result['joints']} | {result['all_checks_pass']} |"
        )
    summary_path = args.output_dir / "RUNTIME_SMOKE_SUMMARY.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.mirror_dir is not None:
        args.mirror_dir.mkdir(parents=True, exist_ok=True)
        for path in (json_path, summary_path):
            (args.mirror_dir / path.name).write_bytes(path.read_bytes())
    print(json.dumps({"status": report["status"], "paths": len(results), "failed": len(failed)}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
