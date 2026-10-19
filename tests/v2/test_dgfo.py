"""Local attention geometry, masking, conservation, and gradient acceptance."""
import unittest

import torch

from mhinet.v2.dgfo import LocalDGFO


class DGFOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(31)

    def test_uniform_attention_uses_only_real_patch_neighbors(self):
        net = LocalDGFO()
        with torch.no_grad():
            for layer in (net.query, net.key):
                layer.weight.zero_()
                layer.bias.zero_()
        features = torch.randn(1, 32, 4, 4)
        flow = torch.arange(32.).reshape(1, 2, 4, 4)
        propagated, weights = net(features, flow, return_attention=True)
        for y in range(4):
            for x in range(4):
                values = flow[0, :, max(0, y-1):y+2, max(0, x-1):x+2]
                torch.testing.assert_close(propagated[0, :, y, x], values.mean((-2, -1)))
        self.assertEqual(int((weights[0, :, 0, 0] > 0).sum()), 4)
        self.assertEqual(int((weights[0, :, 0, 1] > 0).sum()), 6)
        self.assertEqual(int((weights[0, :, 1, 1] > 0).sum()), 9)
        torch.testing.assert_close(weights.sum(1), torch.ones(1, 4, 4))

    def test_masked_values_excluded_and_invalid_queries_retain_flow(self):
        net = LocalDGFO()
        features = torch.randn(1, 32, 4, 4)
        valid = torch.zeros(1, 4, 4, dtype=torch.bool)
        valid[:, 1:3, 1:3] = True
        flow = torch.full((1, 2, 4, 4), 1e6)
        flow[:, :, 1:3, 1:3] = torch.tensor([2., -3.])[None, :, None, None]
        propagated, weights = net(features, flow, valid, return_attention=True)
        torch.testing.assert_close(propagated[:, :, 1:3, 1:3], flow[:, :, 1:3, 1:3])
        torch.testing.assert_close(propagated[:, :, ~valid[0]], flow[:, :, ~valid[0]], rtol=0, atol=0)
        torch.testing.assert_close(weights.sum(1), valid.float())

    def test_constant_zero_and_empty_neighborhoods(self):
        net = LocalDGFO()
        features = torch.randn(2, 32, 4, 4)
        flow = torch.tensor([2., -3.])[None, :, None, None].expand(2, 2, 4, 4).clone()
        torch.testing.assert_close(net(features, flow), flow)
        zero = torch.zeros_like(flow)
        torch.testing.assert_close(net(features, zero), zero, rtol=0, atol=0)
        invalid = torch.zeros(2, 4, 4, dtype=torch.bool)
        propagated, weights = net(features, flow, invalid, return_attention=True)
        torch.testing.assert_close(propagated-flow, zero, rtol=0, atol=0)
        self.assertEqual(int(weights.count_nonzero()), 0)
        self.assertTrue(torch.isfinite(propagated).all())
        empty, attention = net(features[:0], flow[:0], invalid[:0], return_attention=True)
        self.assertEqual(empty.shape, (0, 2, 4, 4))
        self.assertEqual(attention.shape, (0, 9, 4, 4))

    def test_query_key_feature_value_gradients_and_nonfinite_mask(self):
        net = LocalDGFO()
        features = torch.randn(2, 32, 4, 4, requires_grad=True)
        flow = torch.randn(2, 2, 4, 4, requires_grad=True)
        valid = torch.ones(2, 4, 4, dtype=torch.bool)
        valid[:, 0, 0] = False
        out = net(features, flow, valid)
        (out*torch.randn_like(out)).sum().backward()
        for value in (features, flow, net.query.weight, net.key.weight):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0)
        damaged = features.detach().clone()
        damaged[:, :, 1, 1] = float('nan')
        out, weights = net(damaged, flow.detach(), return_attention=True)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(float(weights[:, :, 1, 1].detach().sum()), 0)

    def test_radius_and_shape_contract(self):
        with self.assertRaises(ValueError):
            LocalDGFO(radius=2)
        with self.assertRaises(ValueError):
            LocalDGFO()(torch.zeros(1, 32, 4, 4), torch.zeros(1, 2, 4, 4), torch.ones(1, 4, 4))


if __name__ == '__main__':
    unittest.main()
