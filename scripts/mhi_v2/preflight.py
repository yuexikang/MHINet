#!/usr/bin/env python3
"""Read-only preparation checks for the MHI v2 experiment registry.

This uses only the Python standard library. It never imports model code, loads a
checkpoint, probes a GPU, trains, or evaluates accuracy. The sole output write is
the report path explicitly supplied with --output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo


REPO = Path(__file__).resolve().parents[2]
FREEZE_PATH = "versions/v1/freeze_manifest.json"
PLAN_PATH = "versions/v2/experiment_plan.json"
RUNTIME_PATH = "versions/v1/configs/runtime_snapshot.json"
TRAINING_PATH = "versions/v1/configs/training_snapshot.json"
RELOCATIONS_PATH = "versions/relocations.json"
NEXT_ITEM = "V2-P1-01"
REQUIRED_FILES = (
    "versions/README.md",
    FREEZE_PATH,
    RUNTIME_PATH,
    TRAINING_PATH,
    PLAN_PATH,
    "versions/v2/PLAN.md",
    "versions/v2/ACCEPTANCE.md",
    "versions/v2/proposal.md",
)
STORAGE_PREFIXES = {
    "v1_archive": "versions/v1/",
    "v2_archive": "versions/v2/",
    "v2_configs": "configs/mhi_v2/",
    "v2_scripts": "scripts/mhi_v2/",
    "v2_outputs": "outputs/mhi_v2/",
    "v2_weights": "weight/mhi_v2/",
    "v2_source": "mhinet/v2/",
    "v2_tests": "tests/v2/",
}


def git(*args: str, cwd: Path = REPO) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.decode("utf-8").strip()


def git_blob(ref: str, filename: str) -> bytes:
    path = PurePosixPath(filename)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Frozen source path must be repository-relative: {filename}")
    result = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{ref}:{filename}"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def error_text(error: Exception) -> str:
    if isinstance(error, subprocess.CalledProcessError):
        return error.stderr.decode("utf-8", errors="replace").strip()
    return f"{type(error).__name__}: {error}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_stats(path: Path) -> dict:
    """Hash exact bytes while validating JSON objects one line at a time."""
    digest = hashlib.sha256()
    pairs = 0
    byte_count = 0
    blank_lines = 0
    invalid_rows = 0
    first_invalid_lines = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            digest.update(line)
            byte_count += len(line)
            if not line.strip():
                blank_lines += 1
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Manifest rows must be JSON objects")
            except (ValueError, UnicodeError):
                invalid_rows += 1
                if len(first_invalid_lines) < 5:
                    first_invalid_lines.append(line_number)
            else:
                pairs += 1
    return {
        "sha256": digest.hexdigest(),
        "bytes": byte_count,
        "pairs": pairs,
        "blank_lines": blank_lines,
        "invalid_rows": invalid_rows,
        "first_invalid_lines": first_invalid_lines,
    }


class Preflight:
    def __init__(self, baseline_worktree: Path | None, verify_large_hashes: bool = False):
        self.baseline_worktree = baseline_worktree
        self.verify_large_hashes = verify_large_hashes
        self.resource_hash_cache: dict[tuple[Path, int, int], str] = {}
        self.checks: list[dict] = []
        self.protected_paths = {REPO / path for path in REQUIRED_FILES}
        self.protected_paths.add(REPO / RELOCATIONS_PATH)
        self.protected_paths.add(Path(__file__).resolve())

    def add(self, check_id: str, passed: bool, expected, actual, evidence) -> None:
        self.checks.append({
            "id": check_id,
            "status": "passed" if passed else "failed",
            "expected": expected,
            "actual": actual,
            "evidence": evidence,
        })

    def known(self, check_id: str, expected, actual, evidence) -> None:
        self.checks.append({
            "id": check_id,
            "status": "known_next_task",
            "expected": expected,
            "actual": actual,
            "evidence": evidence,
        })

    def read_json(self, path: str) -> dict | None:
        try:
            value = json.loads((REPO / path).read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Expected a JSON object")
        except (OSError, ValueError, UnicodeError) as error:
            self.add(f"json:{path}", False, "readable JSON object", error_text(error), path)
            return None
        self.add(f"json:{path}", True, "readable JSON object", "readable", path)
        return value

    def check_git(self, freeze: dict, plan: dict) -> None:
        code = freeze["code"]
        tag, commit = code["tag"], code["commit"]
        for check_id, args, expected in (
            ("git:experiment_branch", ("branch", "--show-current"), "codex/mhi-v2"),
            ("git:v1_tag", ("rev-parse", "--verify", f"{tag}^{{commit}}"), commit),
            ("git:main_unchanged", ("rev-parse", "--verify", "refs/heads/main^{commit}"), commit),
        ):
            try:
                actual = git(*args)
                self.add(check_id, actual == expected, expected, actual, list(args))
            except (OSError, subprocess.CalledProcessError) as error:
                self.add(check_id, False, expected, error_text(error), list(args))
        ancestry_args = ("merge-base", "--is-ancestor", commit, "HEAD")
        try:
            git(*ancestry_args)
            self.add("git:baseline_ancestry", True, True, True, list(ancestry_args))
        except subprocess.CalledProcessError as error:
            self.add("git:baseline_ancestry", False, True,
                     {"is_ancestor": False, "exit_code": error.returncode, "error": error_text(error)}, list(ancestry_args))
        except OSError as error:
            self.add("git:baseline_ancestry", False, True, error_text(error), list(ancestry_args))
        self.add(
            "plan:baseline_identity",
            plan["baseline"]["tag"] == tag and plan["baseline"]["commit"] == commit,
            {"tag": tag, "commit": commit},
            {"tag": plan["baseline"]["tag"], "commit": plan["baseline"]["commit"]},
            [FREEZE_PATH, PLAN_PATH],
        )
        for filename, expected in code["tracked_file_sha256"].items():
            try:
                actual = hashlib.sha256(git_blob(commit, filename)).hexdigest()
                self.add(
                    f"v1_tag_source:{filename}", actual == expected, expected, actual,
                    f"git show {commit}:{filename}; current v2 source is not compared",
                )
            except (OSError, ValueError, subprocess.CalledProcessError) as error:
                self.add(f"v1_tag_source:{filename}", False, expected, error_text(error), commit)

    def check_snapshots(self, freeze: dict, runtime: dict | None, training: dict | None) -> None:
        metadata = freeze["checkpoint"]["training_metadata_snapshot"]
        self.add(
            "v1:runtime_snapshot", runtime == metadata["runtime"],
            "Exact checkpoint-embedded runtime object", "matches" if runtime == metadata["runtime"] else "differs or unreadable",
            [RUNTIME_PATH, f"{FREEZE_PATH}:checkpoint.training_metadata_snapshot.runtime"],
        )
        self.add(
            "v1:training_snapshot", training == metadata,
            "Exact checkpoint-embedded training_metadata_snapshot object",
            "matches" if training == metadata else "differs or unreadable",
            [TRAINING_PATH, f"{FREEZE_PATH}:checkpoint.training_metadata_snapshot"],
        )
        gaps = freeze.get("reproduction_gaps", {})
        missing_historical = [item["path"] for item in gaps.get("files", []) if not item.get("present_at_v1_code_tag")]
        self.known(
            "v1:historical_reproduction_gaps",
            "Historical config absence stays documented; snapshot availability does not prove runnable reproduction",
            {"historically_missing_files": missing_historical, "runtime_snapshot_readable": runtime is not None},
            [f"{FREEZE_PATH}:reproduction_gaps", RUNTIME_PATH, TRAINING_PATH],
        )
        self.known(
            "v2:configuration_and_loader", NEXT_ITEM,
            "Versioned configuration parsing, runtime adaptation and strict v1-to-v2 weight migration remain unimplemented",
            "Preparation does not require a v2 model implementation; this report cannot accept P1",
        )
        self.known(
            "v1:evaluator_runtime_adaptation", NEXT_ITEM,
            "The existing v1 evaluator still reads its historical absolute runtime path; archived snapshots are not yet a replay interface",
            "Adapt runtime resolution and verify an actual v1 replay in P1; no GPU replay was performed by preflight",
        )

    def check_archive_integrity(self, freeze: dict) -> None:
        if not (REPO / RELOCATIONS_PATH).is_file():
            return
        relocations = self.read_json(RELOCATIONS_PATH)
        if relocations is None:
            return
        expected = relocations["frozen_manifest_sha256"]
        actual = file_sha256(REPO / FREEZE_PATH)
        self.add("archive:frozen_manifest_bytes", actual == expected, expected, actual,
                 [FREEZE_PATH, f"{RELOCATIONS_PATH}:frozen_manifest_sha256"])
        old_path = freeze["proposal"]["archive"]
        new_path = relocations["paths"][old_path]
        relative = PurePosixPath(new_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Proposal relocation must stay repository-relative")
        expected = freeze["proposal"]["sha256"]
        actual = file_sha256(REPO / new_path)
        self.add("archive:proposal_bytes", actual == expected, expected, actual,
                 {"frozen_reference": old_path, "relocated_path": new_path})

    def check_plan(self, plan: dict) -> None:
        self.add(
            "plan:freeze_manifest_path", plan["baseline"]["freeze_manifest"] == FREEZE_PATH,
            FREEZE_PATH, plan["baseline"]["freeze_manifest"], f"{PLAN_PATH}:baseline.freeze_manifest",
        )
        self.add(
            "plan:locked_evaluation_reference", plan["data_policy"]["locked_evaluation_reference"] == FREEZE_PATH,
            FREEZE_PATH, plan["data_policy"]["locked_evaluation_reference"], f"{PLAN_PATH}:data_policy.locked_evaluation_reference",
        )
        actual_flags = {key: plan.get(key) for key in ("executable", "training_started")}
        self.add(
            "plan:preparation_flags", all(value is False for value in actual_flags.values()),
            {"executable": False, "training_started": False}, actual_flags, PLAN_PATH,
        )
        stages = {stage["id"]: stage["status"] for stage in plan["stages"]}
        expected_stages = {"P0": "completed", **{f"P{number}": "planned" for number in range(1, 6)}}
        self.add("plan:stage_states", stages == expected_stages, expected_stages, stages, f"{PLAN_PATH}:stages")
        experiments = {item["id"]: item["status"] for item in plan["ablation_matrix"]["experiments"]}
        expected_experiments = {f"MHIv2-{letter}": "planned" for letter in "ABCDEF"}
        self.add(
            "plan:ablation_states", experiments == expected_experiments,
            expected_experiments, experiments, f"{PLAN_PATH}:ablation_matrix.experiments",
        )
        storage = plan.get("storage_layout")
        if storage is None:
            self.known("plan:storage_layout", STORAGE_PREFIXES, "not registered", "Register isolation paths before P1")
        else:
            for key, expected in STORAGE_PREFIXES.items():
                self.add(
                    f"plan:storage_layout:{key}", storage.get(key) == expected,
                    expected, storage.get(key), f"{PLAN_PATH}:storage_layout; planned directories need not contain implementations",
                )
        self.add(
            "plan:output_prefix", plan.get("output_prefix") == "outputs/mhi_v2/",
            "outputs/mhi_v2/", plan.get("output_prefix"), f"{PLAN_PATH}:output_prefix",
        )

    def check_manifests(self, freeze: dict, plan: dict) -> None:
        groups = (
            ("training_manifests", {"train", "val"}),
            ("locked_evaluation_manifests", {"GoogleEarth", "JL1Flight"}),
        )
        for group_name, expected_names in groups:
            group = freeze[group_name]
            self.add(f"manifests:{group_name}:inventory", set(group) == expected_names,
                     sorted(expected_names), sorted(group), f"{FREEZE_PATH}:{group_name}")
            for name, entry in group.items():
                path = Path(entry["path"])
                self.protected_paths.add(path.resolve())
                expected = {key: entry[key] for key in ("sha256", "bytes", "pairs")}
                try:
                    actual = manifest_stats(path)
                    passed = actual["invalid_rows"] == 0 and all(actual[key] == value for key, value in expected.items())
                    self.add(f"manifest:{group_name}:{name}", passed, expected, actual, str(path))
                except OSError as error:
                    self.add(f"manifest:{group_name}:{name}", False, expected, error_text(error), str(path))
        expected_paths = {
            "training_manifest": freeze["training_manifests"]["train"]["path"],
            "validation_manifest": freeze["training_manifests"]["val"]["path"],
            "locked_evaluation_manifests": sorted(entry["path"] for entry in freeze["locked_evaluation_manifests"].values()),
        }
        actual_paths = {key: plan["data_policy"].get(key) for key in expected_paths}
        if isinstance(actual_paths["locked_evaluation_manifests"], list):
            actual_paths["locked_evaluation_manifests"] = sorted(actual_paths["locked_evaluation_manifests"])
        self.add("plan:manifest_references", actual_paths == expected_paths, expected_paths, actual_paths, PLAN_PATH)

    def check_resource(self, name: str, entry: dict) -> None:
        path = Path(entry["path"])
        self.protected_paths.add(path.resolve())
        expected = {"bytes": entry["bytes"], "prior_frozen_sha256": entry["sha256"]}
        try:
            stat_before = path.stat()
            actual = {
                "exists": path.is_file(),
                "bytes": stat_before.st_size,
                "verification": "size_only_expected_hash_from_prior_freeze",
                "current_sha256_verified": False,
            }
            passed = actual["exists"] and actual["bytes"] == entry["bytes"]
            if self.verify_large_hashes and actual["exists"]:
                cache_key = (path.resolve(), stat_before.st_size, stat_before.st_mtime_ns)
                if cache_key not in self.resource_hash_cache:
                    self.resource_hash_cache[cache_key] = file_sha256(path)
                actual_hash = self.resource_hash_cache[cache_key]
                stat_after = path.stat()
                stable = (stat_before.st_size, stat_before.st_mtime_ns) == (stat_after.st_size, stat_after.st_mtime_ns)
                actual.update({
                    "verification": "streamed_sha256_and_size",
                    "sha256": actual_hash,
                    "file_stable_during_hash": stable,
                    "current_sha256_verified": actual_hash == entry["sha256"] and stable,
                })
                passed = passed and actual["current_sha256_verified"]
            self.add(f"resource:{name}", passed, expected, actual, str(path))
        except OSError as error:
            self.add(f"resource:{name}", False, expected, error_text(error), str(path))

    def check_worktree(self, freeze: dict) -> None:
        if self.baseline_worktree is None:
            self.checks.append({
                "id": "v1:baseline_worktree", "status": "skipped", "expected": "Optional detached v1 checkout",
                "actual": "--baseline-worktree was not supplied", "evidence": "Does not block preparation acceptance",
            })
            return
        worktree = self.baseline_worktree.resolve()
        expected = freeze["code"]["commit"]
        try:
            actual = git("rev-parse", "--verify", "HEAD^{commit}", cwd=worktree)
            branch = git("branch", "--show-current", cwd=worktree)
            self.add("v1:baseline_worktree", actual == expected and branch == "",
                     {"commit": expected, "branch": ""}, {"commit": actual, "branch": branch}, str(worktree))
        except (OSError, subprocess.CalledProcessError) as error:
            self.add("v1:baseline_worktree", False, {"commit": expected, "branch": ""}, error_text(error), str(worktree))
            return
        for filename, expected_hash in freeze["code"]["tracked_file_sha256"].items():
            if not filename.startswith("mhinet/"):
                continue
            try:
                actual_hash = file_sha256(worktree / filename)
                self.add(f"v1_worktree_source:{filename}", actual_hash == expected_hash,
                         expected_hash, actual_hash, str(worktree / filename))
            except OSError as error:
                self.add(f"v1_worktree_source:{filename}", False, expected_hash, error_text(error), str(worktree / filename))

    def run(self) -> dict:
        present = {path: (REPO / path).is_file() for path in REQUIRED_FILES}
        self.add("archive:required_files", all(present.values()), {path: True for path in REQUIRED_FILES}, present, str(REPO))
        freeze = self.read_json(FREEZE_PATH)
        plan = self.read_json(PLAN_PATH)
        runtime = self.read_json(RUNTIME_PATH)
        training = self.read_json(TRAINING_PATH)
        if freeze is not None and plan is not None:
            operations = (
                ("git_and_baseline", lambda: self.check_git(freeze, plan)),
                ("snapshots", lambda: self.check_snapshots(freeze, runtime, training)),
                ("archive_integrity", lambda: self.check_archive_integrity(freeze)),
                ("plan", lambda: self.check_plan(plan)),
                ("manifests", lambda: self.check_manifests(freeze, plan)),
                ("checkpoint", lambda: self.check_resource("v1_checkpoint", freeze["checkpoint"])),
                ("runtime_resources", lambda: [self.check_resource(name, entry) for name, entry in freeze["runtime_resources"].items()]),
                ("baseline_worktree", lambda: self.check_worktree(freeze)),
            )
            for name, operation in operations:
                try:
                    operation()
                except (KeyError, TypeError, ValueError, OSError) as error:
                    self.add(f"schema:{name}", False, "Complete supported freeze/plan schema", error_text(error), [FREEZE_PATH, PLAN_PATH])
        resource_checks = [check for check in self.checks if check["id"].startswith("resource:")]
        expected_resource_count = (
            1 + len(freeze["runtime_resources"])
            if isinstance(freeze, dict) and isinstance(freeze.get("runtime_resources"), dict) else -1
        )
        all_resources_hashed = len(resource_checks) == expected_resource_count and bool(resource_checks) and all(
            check["status"] == "passed" and isinstance(check["actual"], dict)
            and check["actual"].get("current_sha256_verified") is True
            for check in resource_checks
        )
        if self.verify_large_hashes and all_resources_hashed:
            resource_limitation = "All registered checkpoint/runtime resources passed current streamed SHA256 and byte-size checks. File integrity does not demonstrate checkpoint loading or model behavior."
        elif self.verify_large_hashes:
            resource_limitation = "Current large-file SHA256 verification was requested, but not all registered resource hashes passed. Consult the resource checks; no complete file-integrity verification is claimed."
        else:
            resource_limitation = "Checkpoint/runtime resource availability and byte size were checked. Their current SHA256 was not recomputed; recorded hashes come from the prior freeze."
        return {
            "schema": "mhi.v2-preflight.v1",
            "phase": "preparation_only",
            "status": "failed" if any(check["status"] == "failed" for check in self.checks) else "passed",
            "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            "timezone": "Asia/Shanghai",
            "repository": str(REPO),
            "large_hash_verification_requested": self.verify_large_hashes,
            "checks": self.checks,
            "limitations": [
                "No model code or torch was imported; no GPU, forward-pass, gradient, training or accuracy check ran.",
                resource_limitation,
                "Historical reproduction gaps remain documented. Readable embedded snapshots do not demonstrate a runnable v1 or v2 loader.",
                "The existing v1 evaluator still uses its historical absolute runtime reference; snapshot archival is not a successful GPU replay.",
                "A passed preparation report cannot accept P1 or any later implementation/experiment stage.",
            ],
            "next_item": NEXT_ITEM,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Explicit new JSON report path; existing files are refused by default")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow replacing an existing report file")
    parser.add_argument("--baseline-worktree", type=Path, help="Optional detached mhi-v1 checkout to verify, without changing it")
    parser.add_argument("--verify-large-hashes", action="store_true", help="Recompute checkpoint/runtime-resource SHA256 with bounded-memory streaming; default checks size only")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        parser.error(f"Report already exists: {output}; use a new --output or explicitly pass --overwrite")
    if not output.parent.is_dir():
        parser.error(f"Report parent directory does not exist: {output.parent}")
    preflight = Preflight(args.baseline_worktree, verify_large_hashes=args.verify_large_hashes)
    report = preflight.run()
    if output in {path.resolve() for path in preflight.protected_paths}:
        parser.error(f"Report path is a checked input and cannot be overwritten: {output}")
    try:
        with output.open("w" if args.overwrite else "x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except OSError as error:
        print(f"Cannot write report: {error}", file=sys.stderr)
        return 2
    print(f"MHI v2 preparation: {report['status']}; report: {output}; next: {NEXT_ITEM}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
