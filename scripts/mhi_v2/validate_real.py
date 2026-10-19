"""GPU acceptance on real RGB_512 features; three diagnostic updates, no model save.

Example (run after other GPU acceptance processes finish)::
    python scripts/mhi_v2/validate_real.py --output NEW_EVIDENCE.json --device cuda:0

The two fixed validation rows are descriptive diagnostics only. This script does
not choose hyperparameters or read a sealed evaluation set. GT enters losses and
post-prediction metrics; inference receives only provider outputs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data._utils.collate import default_collate

from mhinet.config import RuntimePaths, sha256_file
from mhinet.downstream.metrics import dense_metrics
from mhinet.downstream.qrru import QRRU
from mhinet.v2.data import RGB512Dataset
from mhinet.v2.losses import flow_head_loss
from mhinet.v2.train import (buffer_digest, implementation_hashes, initialize_system,
                             read_config, seed_all, timestamp, write_json)
from mhinet.v2.training import assert_frozen_gradients, batch_loss


GT_KEYS = {"H_gt_norm", "H_A_to_B", "mask_A_overlap", "mask_B_overlap", "ground_truth"}
MATCH_FIELDS = ("points_a", "points_b", "confidence", "fine_points_b", "lineage")
NEW_BRANCHES = ("diff_encoder.", "diff_adapter.", "dgfo.", "flow_adapter.")


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _emit(event, **fields):
    print(json.dumps(dict(event=event, **fields), ensure_ascii=False, allow_nan=False), flush=True)


def _batch(dataset, index):
    return default_collate([dataset[index]])


def _gpu_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}


def _assert_prediction_inputs(shared):
    leaked = GT_KEYS & set(shared)
    if leaked:
        raise AssertionError(f"GT fields entered prediction: {sorted(leaked)}")


def _assert_result(result, cap):
    count = len(result["confidence"])
    if count > cap:
        raise AssertionError(f"Output count {count} exceeds cap {cap}")
    for name in MATCH_FIELDS:
        value = result[name]
        if len(value) != count or not bool(torch.isfinite(value).all()):
            raise AssertionError(f"Invalid/misaligned output {name}")
    return dict(matches=count, output_cap=cap, before_cap=result.get("output_before_cap"),
                failure_reason=result["failure_reason"])


def _compare_predictions(left, right):
    checked = {}
    for name in MATCH_FIELDS:
        a, b = left[name], right[name]
        if a.shape != b.shape or not torch.equal(a, b):
            raise AssertionError(f"B and C(lambda=0) differ in {name}")
        checked[name] = dict(shape=list(a.shape), exact_equal=True, max_absolute_difference=0.)
    return checked


def _raster_metric(row, root, prediction):
    measured = dense_metrics(row, root, prediction["points_a"].detach().cpu().numpy(),
                              prediction["points_b"].detach().cpu().numpy())
    return dict(matches=measured["matches"], overlap_matches=measured["overlap_matches"],
                pre_at_3=measured["thresholds"]["3.0"]["precision"],
                precision={key: value["precision"] for key, value in measured["thresholds"].items()},
                protocol="raster-mask/GT-H validation diagnostic; not historical GCP precision")


def _trajectory_check(qrru, d2, candidates):
    """Compare the loaded, nontrivial old QRRU with zero-adapter E on real D2."""
    for name in ("diff_adapter", "flow_adapter"):
        if any(bool(torch.count_nonzero(p)) for p in getattr(qrru, name).parameters()):
            raise AssertionError(f"Expected a zero-initialized {name}")
    legacy = QRRU(channels=d2.shape[1], iterations=qrru.iterations, radius=qrru.radius).to(d2.device).eval()
    state = {name: value for name, value in qrru.state_dict().items()
             if not name.startswith(NEW_BRANCHES)}
    legacy.load_state_dict(state, strict=True)
    aa, bb = candidates["fine_all_a"], candidates["fine_all_b"]
    predicted = min(8, len(aa))
    pa = (aa[:predicted]+.5)/2-.5
    pb = (bb[:predicted]+.5)/2-.5
    # This fallback tests numeric equivalence, and makes no claim to be a true
    # correspondence. It uses actual image features, with no GT in inference.
    fallback = d2.new_tensor([[64., 64.], [96., 64.], [128., 64.], [160., 64.],
                             [64., 96.], [96., 96.], [128., 96.], [160., 96.]])
    if predicted < 8:
        pa = torch.cat((pa, fallback[:8-predicted]))
        pb = torch.cat((pb, fallback[:8-predicted]))
    with torch.no_grad():
        old = legacy(d2[0], d2[1], pa, pb)
        new = qrru(d2[0], d2[1], pa, pb)
    differences = {}
    for name in ("flows", "centers", "grids", "gates"):
        delta = (old[name]-new[name]).abs()
        maximum = float(delta.max())
        if not math.isfinite(maximum) or maximum >= 1e-6:
            raise AssertionError(f"Zero adapters changed {name}: {maximum}")
        torch.testing.assert_close(old[name], new[name], rtol=0, atol=1e-6)
        differences[name] = dict(max_absolute_difference=maximum,
                                 per_iteration_max=delta.flatten(1).amax(1).cpu().tolist())
    result = dict(points=8, predicted_points=predicted,
                  fixed_identity_diagnostic_points=8-predicted,
                  fallback_note="Fixed interior equal-coordinate diagnostic only; not a GT match or inference output",
                  source_points_d2=pa.cpu().tolist(), target_points_d2=pb.cpu().tolist(),
                  iterations=qrru.iterations, tolerance=1e-6, trajectories=differences,
                  actual_legacy_control_flow_max=float(old["flows"].abs().max()),
                  new_adapters_all_zero=True)
    del legacy, old, new
    return result


def _gradient_report(qrru):
    prefixes = ("project", "encoder", "gate", "down", "head", "diff_encoder",
                "diff_adapter", "dgfo.query", "dgfo.key", "flow_adapter")
    result = {}
    for prefix in prefixes:
        values = [parameter for name, parameter in qrru.named_parameters()
                  if name == prefix or name.startswith(prefix+".")]
        gradients = [parameter.grad for parameter in values if parameter.grad is not None]
        finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        if not finite:
            raise FloatingPointError(f"Nonfinite {prefix} gradient")
        result[prefix] = dict(parameters=len(values), parameters_with_gradient=len(gradients),
                              parameters_with_nonzero_gradient=sum(bool(torch.count_nonzero(g)) for g in gradients),
                              absolute_gradient_sum=sum(float(g.abs().sum()) for g in gradients),
                              all_gradients_finite=finite)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="New JSON evidence file; existing files are rejected")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/mhi_v2/qrru_e_rgb512_seed42.json")
    parser.add_argument("--runtime", type=Path, help="Optional actual runtime override")
    parser.add_argument("--checkpoint", type=Path, help="Must hash-match the frozen v1 checkpoint")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema="mhi.v2.real-feature-acceptance.v1", status="running", started_at=timestamp(),
                  command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                  argv=sys.argv, cwd=str(Path.cwd()), checks={},
                  scope="Two fixed validation pairs + three diagnostic QRRU updates; no model save or hyperparameter selection")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    started = time.perf_counter()
    hook = None
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))
        config, mc = read_config(args.config)
        if config["stage"] != "qrru" or not (mc.use_discrepancy and mc.use_dgfo):
            raise ValueError("Acceptance requires the E QRRU configuration")
        seed_all(config["seed"])
        runtime_path = args.runtime or Path(config["runtime"])
        runtime = replace(RuntimePaths.from_json(runtime_path), device=args.device)
        freeze_path = ROOT/"versions/v1/freeze_manifest.json"
        freeze = json.loads(freeze_path.read_text())
        source = (args.checkpoint or Path(freeze["checkpoint"]["path"])).resolve()
        actual_source_sha = sha256_file(source)
        if actual_source_sha != freeze["checkpoint"]["sha256"]:
            raise ValueError("Source checkpoint differs from frozen v1 identity")
        actual_hashes = {}
        for path in (runtime.dino_checkpoint, runtime.selected_stage1_checkpoint, runtime.pyramid_checkpoint):
            actual_hashes.setdefault(str(path.resolve()), sha256_file(path))
        report["identities"] = dict(
            frozen_manifest=dict(path=str(freeze_path), sha256=sha256_file(freeze_path)),
            checkpoint=dict(path=str(source), expected_sha256=freeze["checkpoint"]["sha256"],
                            actual_sha256=actual_source_sha, task=freeze["checkpoint"]["task"]),
            initialization_weights_actual_sha256=actual_hashes,
            config=dict(path=str(args.config.resolve()), actual_sha256=sha256_file(args.config)),
            runtime=dict(path=str(runtime_path.resolve()), actual_sha256=sha256_file(runtime_path)))
        report["configuration"] = dict(matcher=asdict(mc), stage="qrru", seed=config["seed"],
                                        learning_rate=config["learning_rate"], diagnostic_steps=3,
                                        predicted_fraction=config.get("predicted_fraction", .5),
                                        correction_limit_d2=config.get("correction_limit_d2", 6.))
        report["environment"] = dict(torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                                      device=str(runtime.device), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
        system, provider, migration = initialize_system(runtime, {**config, "stage": "qrru"}, mc, source)
        system.eval()
        report["provider"] = provider
        report["migration"] = migration
        report["implementation_actual_sha256"] = implementation_hashes(runtime)
        report["implementation_actual_sha256"][str(Path(__file__).resolve())] = sha256_file(__file__)
        report["code_identity_sha256"] = _hash_json(report["implementation_actual_sha256"])
        report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if str(runtime.device).startswith("cuda"):
            report["environment"]["device_name"] = torch.cuda.get_device_name(torch.device(runtime.device))
            torch.cuda.reset_peak_memory_stats()
        fitter_calls = [0]
        def count_fitter(module, inputs, result):
            fitter_calls[0] += 1
        hook = system.shared.stage1_head.homography_fitter.register_forward_hook(count_fitter)
        baseline_buffers = buffer_digest(system)
        val = RGB512Dataset(runtime.data_root/"val/pairs.jsonl", max_pairs=2)
        train = RGB512Dataset(runtime.data_root/"train/pairs.jsonl", max_pairs=3)
        if len(val) != 2 or not len(train):
            raise ValueError("Need two validation pairs and at least one training pair")
        report["data"] = dict(root=str(runtime.data_root), validation_manifest=str(val.manifest),
                               validation_manifest_actual_sha256=sha256_file(val.manifest),
                               train_manifest=str(train.manifest), train_manifest_actual_sha256=sha256_file(train.manifest),
                               grouping_policy=val.grouping_policy,
                               selected_validation_ids=[entry.pair_id for entry in val.index],
                               selected_training_ids=[entry.pair_id for entry in train.index])
        report["validation"] = []
        trajectory_done = False
        # All prediction conditions share this exact E architecture and weights.
        # These labels describe prior switches, not independently trained A-F runs.
        b_config = replace(mc, geometry_mode="none", flow_guidance_enabled=False)
        c_zero_config = replace(mc, geometry_mode="flow_d16", flow_guidance_enabled=True, flow_lambda=0.)
        for index in range(2):
            batch = _batch(val, index)
            tensors = _gpu_batch(batch, runtime.device)
            row = val._read_record(val.index[index].offset)
            with torch.no_grad():
                shared = system.shared(tensors["images"])
                _assert_prediction_inputs(shared)
                shapes = {str(scale): list(value.shape) for scale, value in shared["pyramid"].items()}
                for scale in (8, 4, 2):
                    if shapes[str(scale)] != [1, 2, 256, 512//scale, 512//scale]:
                        raise AssertionError(f"D{scale} shape mismatch: {shapes[str(scale)]}")
                if tuple(shared["flow16_target_norm"].shape) != (1, 2, 32, 32):
                    raise AssertionError("Flow16 shape mismatch")
                if shared["H0_norm"] is not None or shared["stage1_valid"] is not None:
                    raise AssertionError("Flow mode produced fitted H values")
                if shared["call_counts"]["homography_fitter"] != 0 or fitter_calls[0]:
                    raise AssertionError("Flow mode executed an H fitter")
                _, flow_metrics = flow_head_loss(shared["stage1"], tensors["H_gt_norm"],
                                                  tensors["mask_A_overlap"], tensors["mask_B_overlap"])
                system.matcher.config = mc
                candidates = system.matcher.infer(shared, refine=False)[0]
                candidate_summary = _assert_result(candidates, mc.max_output_matches)
                candidate_summary["pre_qrru_population"] = len(candidates["fine_all_a"])
                if not trajectory_done:
                    report["checks"]["real_feature_zero_adapter_trajectory"] = _trajectory_check(
                        system.matcher.qrru, shared["pyramid"][2][0].float(), candidates)
                    trajectory_done = True
                system.matcher.config = b_config
                prediction_b = system.matcher.infer(shared)[0]
                system.matcher.config = c_zero_config
                prediction_c_zero = system.matcher.infer(shared)[0]
                equality = _compare_predictions(prediction_b, prediction_c_zero)
                system.matcher.config = mc
                prediction_c = system.matcher.infer(shared)[0]
                b_summary = _assert_result(prediction_b, mc.max_output_matches)
                zero_summary = _assert_result(prediction_c_zero, mc.max_output_matches)
                c_summary = _assert_result(prediction_c, mc.max_output_matches)
                entry = dict(pair_id=row["pair_id"], source_dataset=row["source_dataset"],
                             source_identity=val.index[index].parent_group,
                             input_shape=list(tensors["images"].shape), pyramid_shapes=shapes,
                             flow_shape=list(shared["flow16_norm"].shape),
                             flow_valid_fraction=float(shared["flow16_valid"].float().mean()),
                             flow_metrics=flow_metrics, call_counts=shared["call_counts"],
                             e_pre_qrru_candidates=candidate_summary,
                             b_vs_c_lambda_zero_exact=equality,
                             b_prior_off=dict(**b_summary, raster_mask_metrics=_raster_metric(row, val.split_root, prediction_b)),
                             c_lambda_zero=zero_summary,
                             c_default_prior=dict(**c_summary, raster_mask_metrics=_raster_metric(row, val.split_root, prediction_c)),
                             comparison_note="Same E weights and D2 features; only prior settings change. Descriptive two-pair validation, no selection.")
                report["validation"].append(entry)
                _emit("validation_pair_passed", pair_id=row["pair_id"], flow_epe_px=flow_metrics["flow_epe_px"],
                      candidates=candidate_summary["pre_qrru_population"],
                      b_pre3=entry["b_prior_off"]["raster_mask_metrics"]["pre_at_3"],
                      c_pre3=entry["c_default_prior"]["raster_mask_metrics"]["pre_at_3"])
            del shared, tensors, candidates, prediction_b, prediction_c_zero, prediction_c
            gc.collect()
        system.matcher.config = mc
        system.configure_stage("qrru")
        system.train()
        # These diagnostic updates never use/pollute the long-run candidate cache.
        diagnostic_config = {**config, "d2_cache": None, "cache_source_identity": ""}
        seed_all(config["seed"])
        optimizer = torch.optim.AdamW(system.optimizer_groups(config["learning_rate"]),
                                      weight_decay=config.get("weight_decay", .0001))
        trainable = [parameter for parameter in system.parameters() if parameter.requires_grad]
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        if optimizer_ids != {id(parameter) for parameter in trainable}:
            raise AssertionError("Optimizer membership differs from trainable parameters")
        frozen_versions = {name: parameter._version for name, parameter in system.named_parameters()
                           if not parameter.requires_grad}
        report["diagnostic_training"] = dict(stage="qrru", batch_size=1, planned_steps=3,
            trainable_parameter_names=system.training_parameter_names(), optimizer_membership_verified=True,
            frozen_buffer_sha256_before=baseline_buffers, model_saved=False, cache_enabled=False,
            gradient_note="Upstream discrepancy/DGFO gradients may be zero initially because adapters start at zero; measured per step, not forced nonzero", steps=[])
        for step in range(1, 4):
            batch = _batch(train, (step-1) % len(train))
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = batch_loss(system, batch, runtime.device, diagnostic_config)
            loss.backward()
            assert_frozen_gradients(system)
            gradients = _gradient_report(system.matcher.qrru)
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
            if not bool(torch.isfinite(gradient_norm)) or float(gradient_norm) == 0:
                raise AssertionError("Expected finite, nonzero QRRU gradients")
            optimizer.step()
            changed_frozen = [name for name, parameter in system.named_parameters()
                              if name in frozen_versions and parameter._version != frozen_versions[name]]
            if changed_frozen:
                raise AssertionError(f"Frozen parameters changed: {changed_frozen[:5]}")
            after_buffers = buffer_digest(system)
            if after_buffers != baseline_buffers:
                raise AssertionError("Frozen buffers changed during QRRU optimization")
            entry = dict(step=step, pair_id=batch["pair_id"][0], loss=float(loss.detach()),
                         gradient_norm_before_clip=float(gradient_norm), gradients=gradients, metrics=metrics,
                         frozen_gradients_absent=True, frozen_parameter_versions_unchanged=True,
                         frozen_buffer_sha256_after=after_buffers)
            report["diagnostic_training"]["steps"].append(entry)
            _emit("diagnostic_training_step_passed", step=step, loss=entry["loss"],
                  gradient_norm=entry["gradient_norm_before_clip"], **metrics)
        if fitter_calls[0]:
            raise AssertionError("Homography fitting ran during real-feature acceptance")
        report["checks"]["actual_homography_fitter_calls"] = fitter_calls[0]
        report["checks"]["output_cap_2048"] = all(
            pair[condition]["matches"] <= 2048 for pair in report["validation"]
            for condition in ("b_prior_off", "c_lambda_zero", "c_default_prior"))
        report["checks"]["frozen_source_checkpoint_unchanged"] = sha256_file(source) == actual_source_sha
        if not report["checks"]["frozen_source_checkpoint_unchanged"]:
            raise AssertionError("Source checkpoint identity changed")
        report["checks"]["no_gt_fields_in_inference"] = True
        report["checks"]["diagnostic_training_steps"] = 3
        report["checks"]["frozen_buffers_unchanged"] = buffer_digest(system) == baseline_buffers
        if str(runtime.device).startswith("cuda"):
            report["environment"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["environment"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["status"] = "passed"
        report["completed_at"] = timestamp()
        report["seconds"] = time.perf_counter()-started
        write_json(output, report)
        _emit("real_feature_acceptance_passed", output=str(output), seconds=report["seconds"])
        return 0
    except BaseException as error:
        report.update(status="failed", completed_at=timestamp(), seconds=time.perf_counter()-started,
                      error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        write_json(output, report)
        raise
    finally:
        if hook is not None:
            hook.remove()


if __name__ == "__main__":
    raise SystemExit(main())
