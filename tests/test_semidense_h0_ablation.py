"""Inference-only H0 ablations against the frozen be04d1f baseline."""
import copy
from dataclasses import asdict
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import mhinet.downstream.semidense as matching
from mhinet.downstream.semidense import SemidenseConfig, SemidenseMatcher
from mhinet.downstream.supervision import fine_grids, unwarped_fine_grids, to_uv

# This historical infer function is intentional: A must preserve the original
# numerical path, including diagnostics, ranking, confidence, and QRRU filtering.
LEGACY_INFER_SOURCE = r'''
@torch.no_grad()
def infer(self,shared,sizes=None,*,diagnostics=False):
    """No GT argument. Returns one variable-length match set per image pair."""
    results=[];c=self.config
    hc=(c.input_size//8,)*2;hf=(c.input_size//2,)*2
    for batch in range(len(shared['H0_norm'])):
        d8=shared['pyramid'][8][batch].float();d2=shared['pyramid'][2][batch].float()
        H=shared['H0_norm'][batch].float()
        size=sizes[batch] if sizes is not None else ((c.input_size,c.input_size),(c.input_size,c.input_size))
        empty=d2.new_empty((0,2))
        if not bool(shared['stage1_valid'][batch]) or not legal_h(H):
            results.append(dict(points_a=empty,points_b=empty,confidence=empty[:,0],failure_reason='invalid_h0'));continue
        if d8.shape != (2,256,*hc) or d2.shape != (2,256,*hf):
            raise ValueError(f'Inference requires registered {c.input_size}/D8/D2 feature contract')
        masks=predicted_overlap_masks(H[None],hc[0])
        coarse=match_d8(d8[0],d8[1],masks.a,masks.b,temperature=float(self.tau_c),
            threshold=c.confidence_threshold,max_coarse=c.max_coarse,policy='mutual',
            chunk_rows=c.chunk,force_chunked=True)
        mask2=predicted_overlap_masks(H[None],hf[0])
        matches=[];fine_trace=[];qrru_trace=[]
        probe_parents=set(torch.linspace(0,max(0,len(coarse.source_flat)-1),4).long().tolist())
        for start in range(0,len(coarse.source_flat),c.inference_window_chunk):
            end=start+c.inference_window_chunk
            ga,gb=fine_grids(coarse.source_flat[start:end],coarse.target_flat[start:end],H,hc,hf)
            va=valid_points(mask2.a,ga);vb=valid_points(mask2.b,gb)
            lp=self.fine_scores(d2[0],d2[1],ga,gb,va,vb)
            if diagnostics and len(fine_trace)<4:
                for k in range(len(ga)):
                    if start+k not in probe_parents:continue
                    fine_trace.append(dict(parent=start+k,a=ga[k].cpu(),b=gb[k].cpu(),
                        logp=lp[k].cpu(),valid_a=va[k].cpu(),valid_b=vb[k].cpu()))
            best=lp.argmax(-1);reverse=lp.argmax(-2)
            ids=torch.arange(16,device=d2.device)[None].expand_as(best)
            valid=va & vb.gather(1,best) & (reverse.gather(1,best)==ids)
            m,s=valid.nonzero(as_tuple=True);t=best[m,s]
            score=(lp[m,s,t].exp()*coarse.confidence[start:end][m]).sqrt()
            keep=score>=c.confidence_threshold
            matches.append((ga[m[keep],s[keep]],gb[m[keep],t[keep]],score[keep],torch.stack((m[keep]+start,s[keep],t[keep]),-1)))
        if not matches or sum(len(x[0]) for x in matches)==0:
            results.append(dict(points_a=empty,points_b=empty,confidence=empty[:,0],failure_reason='no_fine_matches'));continue
        a,b,score,lineage=[torch.cat([x[k] for x in matches]) for k in range(4)]
        order=score.argsort(descending=True,stable=True)[:c.max_matches]
        a,b,score,lineage=a[order],b[order],score[order],lineage[order]
        projected=self.qrru.project(d2)
        refined=[];out_of_bounds=0
        for start in range(0,len(a),c.inference_window_chunk):
            end=start+c.inference_window_chunk
            qr=self.qrru(projected[0],projected[1],to_uv(a[start:end],hf),to_uv(b[start:end],hf),projected=True)
            if diagnostics and start==0:
                qrru_trace.append(dict(a=a[start:end][:8].cpu(),b=b[start:end][:8].cpu(),lineage=lineage[start:end][:8].cpu(),
                    **{k:v[:,:8].cpu() for k,v in qr.items()}))
            refined.append(to_norm(qr['centers'][-1],hf))
            grid=qr['grids'][-1]
            out_of_bounds+=int(((grid<0)|(grid>grid.new_tensor([hf[1]-1,hf[0]-1]))).any(-1).sum())
        out=torch.cat(refined)
        valid=torch.isfinite(out).all(-1)&(out>=-1).all(-1)&(out<=1).all(-1)
        results.append(dict(points_a=centers_to_native(a[valid],size[0]),points_b=centers_to_native(out[valid],size[1]),
            fine_points_b=centers_to_native(b[valid],size[1]),confidence=score[valid],
            fine_all_a=centers_to_native(a,size[0]),fine_all_b=centers_to_native(b,size[1]),
            coarse_a=centers_to_native(flat_indices_to_centers(coarse.source_flat,*hc),size[0]),
            coarse_b=centers_to_native(flat_indices_to_centers(coarse.target_flat,*hc),size[1]),
            qrru_outside_samples=out_of_bounds,rejected_qrru=int((~valid).sum()),
            failure_reason='none' if bool(valid.any()) else 'invalid_qrru_output'))
        if diagnostics:
            results[-1]['diagnostics']=dict(coarse_source=coarse.source_flat.cpu(),coarse_target=coarse.target_flat.cpu(),
                coarse_confidence=coarse.confidence.cpu(),coarse_counts=coarse.diagnostics,
                mask_a=masks.a.cpu(),mask_b=masks.b.cpu(),fine=fine_trace,qrru=qrru_trace,
                fine_before_cap=sum(len(x[0]) for x in matches),fine_after_cap=len(a),lineage=lineage.cpu(),
                qrru_valid=valid.cpu())
    return results

'''
_legacy_namespace = {}
_legacy_globals = matching.__dict__.copy()
exec(LEGACY_INFER_SOURCE, _legacy_globals, _legacy_namespace)
legacy_infer = _legacy_namespace["infer"]

VARIANTS = ((True, True), (True, False), (False, True), (False, False))


class H0InferenceAblationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(21)
        self.matcher = SemidenseMatcher(SemidenseConfig(
            input_size=512, iterations=2, max_matches=32))
        # A nonzero learned update exercises QRRU instead of only its initial identity.
        torch.nn.init.normal_(self.matcher.qrru.head[-1].weight, std=.02)
        self.matcher.eval()
        self.shared = dict(
            pyramid={
                8: torch.randn(1, 2, 1, 64, 64).expand(-1, -1, 256, -1, -1),
                2: torch.randn(1, 2, 1, 256, 256).expand(-1, -1, 256, -1, -1)},
            H0_norm=torch.tensor([[[1.10, .08, .12],
                                   [-.04, .90, -.04],
                                   [.01, -.01, 1.]]]),
            stage1_valid=torch.tensor([True]))
        self.coarse = SimpleNamespace(
            source_flat=torch.tensor([1300, 2000]),
            target_flat=torch.tensor([1365, 2065]),
            confidence=torch.tensor([.7, .8]),
            diagnostics=dict(selected=2))
        self.sizes = [((600, 400), (720, 480))]

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected),
                            f"Tensor changed: {tuple(expected.shape)}")
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(actual, expected)

    def infer(self, options=(True, True), shared=None, diagnostics=True):
        with patch.object(matching, "match_d8", return_value=self.coarse):
            return self.matcher.infer(
                self.shared if shared is None else shared, self.sizes,
                diagnostics=diagnostics,
                use_h0_d8_mask=options[0],
                use_h0_d2_warp=options[1])

    def test_a_is_bitwise_equal_to_historical_infer(self):
        with patch.object(matching, "match_d8", return_value=self.coarse) as coarse, \
             patch.dict(_legacy_globals, match_d8=coarse):
            expected = legacy_infer(
                self.matcher, self.shared, self.sizes, diagnostics=True)
            default = self.matcher.infer(
                self.shared, self.sizes, diagnostics=True)
        explicit = self.infer()
        self.assertEqual(expected[0]["failure_reason"], "none")
        self.assert_nested_equal(default, expected)
        self.assert_nested_equal(explicit, expected)

    def test_each_variant_uses_only_its_registered_h0_positions(self):
        original = matching.predicted_overlap_masks
        for use_d8, use_d2 in VARIANTS:
            with self.subTest(d8=use_d8, d2=use_d2), \
                 patch.object(matching, "predicted_overlap_masks", wraps=original) as masks, \
                 patch.object(matching, "fine_grids", wraps=fine_grids) as warped, \
                 patch.object(matching, "unwarped_fine_grids",
                              wraps=unwarped_fine_grids) as unwarped:
                result = self.infer((use_d8, use_d2))[0]
            self.assertEqual([call.args[1] for call in masks.call_args_list],
                             ([64] if use_d8 else []) +
                             ([256] if use_d2 else []))
            self.assertEqual(warped.call_count, int(use_d2))
            self.assertEqual(unwarped.call_count, int(not use_d2))
            diag = result["diagnostics"]
            if use_d8:
                self.assertFalse(bool(diag["mask_a"].all()))
            else:
                self.assertTrue(bool(diag["mask_a"].all()))
                self.assertTrue(bool(diag["mask_b"].all()))
            expected_a, expected_b = (
                fine_grids(self.coarse.source_flat, self.coarse.target_flat,
                           self.shared["H0_norm"][0], (64, 64), (256, 256))
                if use_d2 else
                unwarped_fine_grids(self.coarse.source_flat,
                                   self.coarse.target_flat,
                                   (64, 64), (256, 256)))
            for fine in diag["fine"]:
                parent = fine["parent"]
                self.assertTrue(torch.equal(fine["a"], expected_a[parent]))
                self.assertTrue(torch.equal(fine["b"], expected_b[parent]))

    def test_regular_children_are_exact_target_d8_cell_pixels(self):
        for coarse_hw in ((8, 8), (64, 64), (98, 98)):
            width = coarse_hw[1]
            fine_hw = (coarse_hw[0]*4, width*4)
            i = torch.tensor([width+1, 2*width+2])
            j = torch.tensor([3*width+4, 4*width+5])
            a, b = unwarped_fine_grids(i, j, coarse_hw, fine_hw)
            oy, ox = torch.meshgrid(torch.arange(4), torch.arange(4),
                                    indexing="ij")
            for indices, lattice in ((i, a), (j, b)):
                expected = torch.stack(
                    ((indices%width)[:, None]*4+ox.flatten(),
                     (indices//width)[:, None]*4+oy.flatten()), -1).float()
                torch.testing.assert_close(to_uv(lattice, fine_hw), expected,
                                           atol=4e-5, rtol=0)
        with self.assertRaisesRegex(ValueError, "scale ratio"):
            unwarped_fine_grids(i, j, (8, 8), (31, 32))

    def test_d_has_no_dependency_on_h0_or_its_validity(self):
        expected = self.infer((False, False))
        variants = [
            dict(pyramid=self.shared["pyramid"]),
            dict(self.shared, H0_norm=None, stage1_valid=None),
            dict(self.shared, H0_norm=torch.zeros(1, 3, 3),
                 stage1_valid=torch.tensor([False])),
            dict(self.shared, H0_norm=torch.full((1, 3, 3), float("nan")),
                 stage1_valid=torch.tensor([False])),
            dict(self.shared, H0_norm=torch.eye(3)[None]*1000,
                 stage1_valid=torch.tensor([True]))]
        for shared in variants:
            with patch.object(matching, "predicted_overlap_masks",
                              side_effect=AssertionError("H0 unexpectedly used")), \
                 patch.object(matching, "legal_h",
                              side_effect=AssertionError("H0 unexpectedly checked")):
                self.assert_nested_equal(self.infer((False, False), shared),
                                         expected)

    def test_guided_variants_keep_invalid_h0_failure(self):
        shared = dict(self.shared, H0_norm=torch.zeros(1, 3, 3),
                      stage1_valid=torch.tensor([False]))
        for options in VARIANTS[:3]:
            result = self.infer(options, shared)[0]
            self.assertEqual(result["failure_reason"], "invalid_h0")
            self.assertEqual(len(result["points_a"]), 0)

    def test_switches_do_not_modify_weights_buffers_or_training_config(self):
        state = {key: value.clone() for key, value in
                 self.matcher.state_dict().items()}
        config = copy.deepcopy(asdict(self.matcher.config))
        for options in VARIANTS:
            self.infer(options)
        self.assert_nested_equal(self.matcher.state_dict(), state)
        self.assertEqual(asdict(self.matcher.config), config)
        restored = SemidenseMatcher(SemidenseConfig(**config))
        restored.load_state_dict(self.matcher.state_dict(), strict=True)
        self.assertEqual(restored.state_dict().keys(), state.keys())

    def test_switches_must_be_explicit_booleans(self):
        with self.assertRaisesRegex(TypeError, "must be bool"):
            self.matcher.infer(self.shared, use_h0_d8_mask="false")
        with self.assertRaisesRegex(TypeError, "must be bool"):
            self.matcher.infer(self.shared, use_h0_d2_warp=0)


if __name__ == "__main__":
    unittest.main()
