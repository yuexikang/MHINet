"""Independent acceptance checks: paired gains and the shared D8 parent budget."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mhinet.v2.flow_loss.acceptance import metric_gates, selected_parent_ids, parent_retention
from mhinet.v2.flow_loss.common import digest_tree


class AcceptanceTests(unittest.TestCase):
    def test_engineering_artifacts_require_exact_replay_and_identical_pair_order(self):
        import torch
        from mhinet.config import sha256_file
        from mhinet.v2.flow_loss import acceptance
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            out = Path(directory)
            keys = [f"head.{i}" for i in range(10)]
            model = {k:torch.zeros(1) for k in keys+["frozen"]}
            source = out/"source.pt"; torch.save(dict(model=model), source)
            (out/"calibration").mkdir()
            (out/"calibration/calibration.json").write_text('{}')
            config = dict(arm="L0", order_seed=47)
            metadata = dict(source_checkpoint=str(source), source_sha256="source", train_sha256="train",
                val_sha256="val", trainable_parameters=keys, flow_loss_pilot=dict(
                    source_state_clone={"every_source_tensor_exact":True}, config=config,
                    sixth_pass_order_sha256="order", sixth_pass_first16_pair_ids=["a","b"]))
            state = dict(model={**model, keys[0]:torch.ones(1)}, optimizer={}, scheduler={"completed":2},
                progress={"optimizer_step":25537}, rng={"cpu":torch.arange(4)}, auxiliary_state={}, metadata=metadata)
            for name in ("L0_continuous", "L0_resumed", "L1"):
                target=out/"engineering"/name; target.mkdir(parents=True)
                arm=copy.deepcopy(state)
                if name == "L1":
                    arm["metadata"]["flow_loss_pilot"]["config"]["arm"]="L1"
                    arm["progress"]["optimizer_step"]=25536
                torch.save(arm,target/"latest.pt")
                (target/"run.json").write_text(json.dumps(arm["metadata"]))
                (target/"train.jsonl").write_text(json.dumps({"pair_ids":["a"]})+'\n'+
                    (json.dumps({"pair_ids":["b"]})+'\n' if name != "L1" else ''))
            with patch.object(acceptance,"SOURCE_SHA",sha256_file(source)), patch("builtins.print"):
                self.assertEqual(acceptance.engineering(out),0)
                # The actual saved optimizer/RNG state is compared, independently of logs.
                broken=copy.deepcopy(state); broken["rng"]["cpu"][0]=42
                torch.save(broken,out/"engineering/L0_resumed/latest.pt")
                with self.assertRaisesRegex(RuntimeError,"rng"):
                    acceptance.engineering(out)
                torch.save(state,out/"engineering/L0_resumed/latest.pt")
                (out/"engineering/L1/train.jsonl").write_text(json.dumps({"pair_ids":["wrong"]})+'\n')
                with self.assertRaisesRegex(RuntimeError,"paired first update"):
                    acceptance.engineering(out)

    def test_matching_gain_does_not_hide_loss_of_coverage_count_or_failures(self):
        base = dict(stages=dict(final=dict(macro_raster_mask_precision={"3.0": .8},
            mean_coverage_a_16x16=.6, mean_coverage_b_16x16=.5,
            mean_matches=1800, failure_below20_correct5_rate=.01)))
        candidate = copy.deepcopy(base)
        candidate["stages"]["final"]["macro_raster_mask_precision"]["3.0"] = .803
        self.assertTrue(all(metric_gates(base, candidate, .76, .75, True).values()))
        for field, value, gate in (("mean_matches", 1700, "match_count_preserved"),
                ("mean_coverage_b_16x16", .48, "coverage_b_preserved"),
                ("failure_below20_correct5_rate", .016, "failure_increase_at_most_point5pp")):
            wrong = copy.deepcopy(candidate); wrong["stages"]["final"][field] = value
            self.assertFalse(metric_gates(base, wrong, .76, .75, True)[gate])
        self.assertFalse(metric_gates(base, candidate, .76, .77, True)["flow_epe_improved"])
        self.assertFalse(metric_gates(base, candidate, .76, .75, False)["prior_off_exact"])

    def test_d8_native_centers_recover_exact_parent_cells(self):
        np.testing.assert_array_equal(selected_parent_ids(np.array([[3.5,3.5], [507.5,507.5], [11.5,19.5]])),
                                      [0,4095,129])
        for invalid in (np.array([[0.,0.]]), np.array([[515.5,3.5]]), np.array([[np.nan,3.5]])):
            with self.assertRaises(ValueError):
                selected_parent_ids(invalid)

    def test_expected_matching_failure_keeps_missing_d8_evidence_explicit(self):
        import torch
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory); Image.fromarray(np.full((16,16),255,dtype=np.uint8)).save(root/"mask.png")
            row=dict(H_A_to_B=np.eye(3).tolist(), mask_A_overlap="mask.png", mask_B_overlap="mask.png")
            predictions={"missing":dict(d8_a=None,d8_b=None),
                         "available":dict(d8_a=np.array([[3.5,3.5]]),d8_b=np.array([[3.5,3.5]]))}
            with patch("mhinet.v2.flow_loss.acceptance.coarse_labels",return_value=(torch.tensor([0,1]),torch.tensor([0,1]))):
                result=parent_retention(row,root,predictions)
            self.assertFalse(result["missing"]["available"])
            self.assertIsNone(result["missing"]["recall"])
            self.assertEqual(result["missing"]["gt_parents"],2)
            self.assertEqual(result["available"]["recall"],.5)

    def test_durable_queue_checks_hidden_gpu_occupancy_and_proof_precedes_training(self):
        path = Path(__file__).resolve().parents[3]/"scripts/mhi_v2/run_flow_loss_experiment.py"
        spec = importlib.util.spec_from_file_location("flow_loss_queue_test", path)
        queue = importlib.util.module_from_spec(spec); spec.loader.exec_module(queue)
        for response, available in (("16131, 0", False), ("0, 91", False), ("42, 0", True)):
            with patch.object(queue.subprocess, "check_output", return_value=response):
                self.assertEqual(queue.gpu_status(2)["available"], available)
        reg = dict(output="/experiment", manifest="/val", source_checkpoint="/source", runtime="/runtime",
                   base_config="/base", hard_cases="/hard", configs={"L0":"/L0", "L1":"/L1"})
        stages = queue.build_commands(reg); names = [s["name"] for s in stages]
        self.assertLess(names.index("engineering_acceptance"), names.index("train_L0"))
        self.assertLess(names.index("train_L1"), names.index("flow_L1"))
        self.assertEqual([s["name"] for s in stages if s["name"].startswith("matching_")],
                         ["matching_L0_B", "matching_L0_C", "matching_L1_B", "matching_L1_C"])
        self.assertTrue(all(s["full"] for s in stages if s["name"].startswith(("matching_", "flow_L", "flow_source"))))

    def test_tree_digest_includes_type_shape_and_values(self):
        import torch
        self.assertNotEqual(digest_tree(torch.ones(2, dtype=torch.bfloat16)), digest_tree(torch.ones(2)))
        self.assertNotEqual(digest_tree(torch.ones(2)), digest_tree(torch.ones(1,2)))
        self.assertNotEqual(digest_tree([1,2]), digest_tree((1,2)))


if __name__ == "__main__":
    unittest.main()
