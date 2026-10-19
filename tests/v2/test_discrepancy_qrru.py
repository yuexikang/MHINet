"""Verify old-weight equivalence, real sampling masks, and learned branches."""
import unittest

import torch

from mhinet.downstream.qrru import QRRU
from mhinet.downstream.supervision import offsets, interpolate_controls
from mhinet.v2.qrru import QRRUV2


class DiscrepancyQRRUTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(23)

    def test_disabled_registers_exact_legacy_parameters(self):
        old = QRRU(channels=8)
        new = QRRUV2(channels=8)
        self.assertEqual({k: v.shape for k, v in old.state_dict().items()},
                         {k: v.shape for k, v in new.state_dict().items()})
        new.load_state_dict(old.state_dict(), strict=True)
        self.assertEqual(new.encoder[0].in_channels, 128)
        with self.assertRaises(ValueError):
            QRRUV2(use_dgfo=True)

    def test_zero_initialized_adapters_match_all_nontrivial_legacy_trajectories(self):
        old = QRRU(channels=8)
        # A trained-style nonzero head makes this a trajectory comparison rather
        # than the trivial identity output of a newly initialized old model.
        with torch.no_grad():
            old.head[-1].weight.normal_(0, .04)
            old.head[-1].bias.normal_(0, .03)
        fa, fb = torch.randn(8, 20, 20), torch.randn(8, 20, 20)
        pa = torch.tensor([[8., 8.], [0., 0.], [19., 19.], [-20., -20.]])
        pb = pa+torch.tensor([.6, -.4])
        expected = old(fa, fb, pa, pb)
        self.assertGreater(float(expected['flows'].detach().abs().sum()), 0)
        for discrepancy, dgfo in ((False, False), (True, False), (True, True)):
            with self.subTest(discrepancy=discrepancy, dgfo=dgfo):
                new = QRRUV2(channels=8, use_discrepancy=discrepancy, use_dgfo=dgfo)
                loaded = new.load_state_dict(old.state_dict(), strict=False)
                self.assertEqual(loaded.unexpected_keys, [])
                self.assertTrue(all(k.startswith(('diff_', 'dgfo.', 'flow_adapter.')) for k in loaded.missing_keys))
                for name, value in old.state_dict().items():
                    torch.testing.assert_close(new.state_dict()[name], value, rtol=0, atol=0)
                actual = new(fa, fb, pa, pb)
                projected = new.project(torch.stack((fa, fb)))
                actual_projected = new(projected[0], projected[1], pa, pb, projected=True)
                self.assertEqual(actual.keys(), expected.keys())
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                    torch.testing.assert_close(actual_projected[key], expected[key], rtol=1e-5, atol=1e-6)

    def test_real_source_warped_target_mask_and_original_control_extrapolation(self):
        net = QRRUV2(channels=8, use_discrepancy=True, use_dgfo=True)
        with torch.no_grad():
            net.head[-1].weight.normal_(0, .08)
        fa, fb = torch.randn(8, 8, 8), torch.randn(8, 6, 7)
        pa = torch.tensor([[0., 0.], [4., 4.], [7., 7.], [-20., -20.]])
        pb = pa+torch.tensor([.6, -.4])
        captured = []
        def record(module, args, output):
            captured.append(tuple(x.detach().clone() for x in (args[1], args[2], output)))
        hook = net.dgfo.register_forward_hook(record)
        result = net(fa, fb, pa, pb)
        hook.remove()
        off = offsets('cpu')
        ga = pa[:, None, None]+off
        gb0 = pb[:, None, None]+off
        for t, (local_flow, valid, propagated) in enumerate(captured):
            controls = torch.zeros(len(pa), 2, 2, 2) if t == 0 else result['flows'][t-1]
            expected = interpolate_controls(controls, off)
            torch.testing.assert_close(local_flow.permute(0, 2, 3, 1), expected)
            gb = gb0+expected
            source_ok = (ga >= 0).all(-1) & (ga <= torch.tensor([7, 7])).all(-1)
            target_ok = (gb >= 0).all(-1) & (gb <= torch.tensor([6, 5])).all(-1)
            torch.testing.assert_close(valid, source_ok & target_ok)
            torch.testing.assert_close(propagated[-1], local_flow[-1], rtol=0, atol=0)
            if t == 0:
                torch.testing.assert_close(propagated-local_flow, torch.zeros_like(local_flow), rtol=0, atol=0)
        self.assertEqual(net.flow_adapter.bias, None)
        # The corner's +/-1.5 sample extrapolates beyond the +/-1 controls.
        controls = torch.zeros(1, 2, 2, 2)
        controls[0, 0] = torch.tensor([[-1., 1.], [-1., 1.]])
        torch.testing.assert_close(interpolate_controls(controls, off)[0, :, :, 0], off[:, :, 0])

    def test_upstream_and_attention_gradients_after_zero_adapter_warmup(self):
        net = QRRUV2(channels=8, use_discrepancy=True, use_dgfo=True)
        optimizer = torch.optim.Adam(net.parameters(), lr=.005)
        fa = torch.randn(8, 20, 20, requires_grad=True)
        fb = torch.randn(8, 20, 20, requires_grad=True)
        pa = torch.tensor([[4., 4.], [8., 8.], [12., 12.], [16., 16.]])
        pb = pa+torch.tensor([.5, -.25])
        target = torch.randn(4, 2, 2, 2)*.2
        target[:, 0] += .4
        for _ in range(6):
            optimizer.zero_grad()
            fa.grad = fb.grad = None
            result = net(fa, fb, pa, pb)
            loss = (result['flows']-target[None]).square().mean()
            loss.backward()
            optimizer.step()
        values = (fa, fb, net.project.weight, net.diff_encoder[0].weight,
                  net.diff_adapter.weight, net.flow_adapter.weight,
                  net.dgfo.query.weight, net.dgfo.key.weight)
        for value in values:
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0)

    def test_empty_matches_all_variants_have_finite_shapes_and_backward(self):
        for discrepancy, dgfo in ((False, False), (True, False), (True, True)):
            net = QRRUV2(channels=8, use_discrepancy=discrepancy, use_dgfo=dgfo)
            fa = torch.randn(8, 6, 6, requires_grad=True)
            fb = torch.randn_like(fa, requires_grad=True)
            points = torch.empty(0, 2)
            result = net(fa, fb, points, points)
            self.assertEqual(result['flows'].shape, (4, 0, 2, 2, 2))
            self.assertEqual(result['centers'].shape, (4, 0, 2))
            self.assertEqual(result['grids'].shape, (4, 0, 4, 4, 2))
            self.assertEqual(result['gates'].shape, (4, 0, 1, 2, 2))
            sum(value.sum() for value in result.values()).backward()
            self.assertTrue(torch.isfinite(fa.grad).all())
            self.assertTrue(torch.isfinite(fb.grad).all())


if __name__ == '__main__':
    unittest.main()
