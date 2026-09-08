"""Unit tests for the parameterized MHINet v1 refinement modules."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from mhinet.modules import (
    EXPECTED_NEW_PARAMETERS,
    SCALE_SPECS,
    DownBlock,
    PyramidAdapter,
    ResidualGeometryDecoder,
    build_multiscale_modules,
    count_new_parameters,
)


class PyramidAdapterTests(unittest.TestCase):
    def test_pair_shape_normalization_and_zero_norm_mask(self) -> None:
        torch.manual_seed(7)
        adapter = PyramidAdapter(32)
        descriptors = torch.randn(1, 2, 256, 3, 5)
        descriptors[0, 0, :, 1, 2] = 0

        features, valid = adapter(descriptors)

        self.assertEqual(features.shape, (1, 2, 32, 3, 5))
        self.assertEqual(valid.shape, (1, 2, 1, 3, 5))
        self.assertEqual(valid.dtype, torch.bool)
        self.assertFalse(bool(valid[0, 0, 0, 1, 2]))
        self.assertTrue(torch.equal(features[0, 0, :, 1, 2], torch.zeros(32)))
        norms = torch.linalg.vector_norm(features, dim=-3, keepdim=True)
        torch.testing.assert_close(
            norms[valid],
            torch.ones_like(norms[valid]),
            atol=2e-5,
            rtol=2e-5,
        )

    def test_adapter_is_bias_free_and_kaiming_nonzero(self) -> None:
        adapter = PyramidAdapter(64)
        self.assertIsNone(adapter.projection.bias)
        self.assertGreater(int(torch.count_nonzero(adapter.projection.weight)), 0)

    def test_adapter_rejects_nonfinite_input(self) -> None:
        adapter = PyramidAdapter(32)
        descriptors = torch.zeros(1, 256, 2, 2)
        descriptors[0, 0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            adapter(descriptors)


class DecoderStructureTests(unittest.TestCase):
    def test_down_block_contract_and_parameter_count(self) -> None:
        block = DownBlock()
        self.assertEqual(sum(parameter.numel() for parameter in block.parameters()), 78_080)
        output = block(torch.randn(2, 64, 13, 17))
        self.assertEqual(output.shape, (2, 64, 7, 9))

    def test_factory_shapes_initialization_and_parameter_count(self) -> None:
        adapters, decoders = build_multiscale_modules()
        self.assertEqual(tuple(adapters.keys()), ("8", "4", "2", "1"))
        self.assertEqual(tuple(decoders.keys()), ("8", "4", "2", "1"))
        self.assertEqual(count_new_parameters(adapters, decoders), EXPECTED_NEW_PARAMETERS)

        expected_counts = {
            8: (16_384, 509_448),
            4: (16_384, 587_528),
            2: (8_192, 661_512),
            1: (8_192, 736_520),
        }
        for scale, spec in SCALE_SPECS.items():
            adapter = adapters[str(scale)]
            decoder = decoders[str(scale)]
            self.assertEqual(
                spec.decoder_input_channels,
                2 * spec.correlation_candidates + 4,
            )
            self.assertEqual(adapter.out_channels, spec.adapter_channels)
            self.assertEqual(decoder.input_channels, spec.decoder_input_channels)
            self.assertEqual(len(decoder.down_blocks), spec.down_blocks)
            self.assertEqual(decoder.max_delta_px, spec.max_delta_px)
            self.assertEqual(
                sum(parameter.numel() for parameter in adapter.parameters()),
                expected_counts[scale][0],
            )
            self.assertEqual(
                sum(parameter.numel() for parameter in decoder.parameters()),
                expected_counts[scale][1],
            )
            self.assertTrue(torch.equal(decoder.fc2.weight, torch.zeros_like(decoder.fc2.weight)))
            self.assertTrue(torch.equal(decoder.fc2.bias, torch.zeros_like(decoder.fc2.bias)))

            for module in decoder.modules():
                if isinstance(module, nn.GroupNorm):
                    self.assertTrue(torch.equal(module.weight, torch.ones_like(module.weight)))
                    self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))
                if isinstance(module, (nn.Conv2d, nn.Linear)) and module is not decoder.fc2:
                    self.assertGreater(int(torch.count_nonzero(module.weight)), 0)
                    if module.bias is not None:
                        self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))

    def test_full_contract_shapes_on_meta_device(self) -> None:
        for scale, spec in SCALE_SPECS.items():
            spatial = spec.spatial_size
            adapter = PyramidAdapter(spec.adapter_channels).to(device="meta")
            descriptors = torch.empty(
                1,
                2,
                256,
                spatial,
                spatial,
                device="meta",
            )
            features, valid = adapter(descriptors)
            self.assertEqual(
                features.shape,
                (1, 2, spec.adapter_channels, spatial, spatial),
                msg=f"adapter scale D{scale}",
            )
            self.assertEqual(
                valid.shape,
                (1, 2, 1, spatial, spatial),
                msg=f"adapter mask scale D{scale}",
            )

            decoder = ResidualGeometryDecoder(
                spec.decoder_input_channels,
                spec.down_blocks,
                spec.max_delta_px,
            ).to(device="meta")
            decoder_input = torch.empty(
                1,
                spec.decoder_input_channels,
                spatial,
                spatial,
                device="meta",
            )
            encoded = decoder.stem_activation(
                decoder.stem_norm(decoder.stem_conv(decoder_input))
            )
            for block in decoder.down_blocks:
                encoded = block(encoded)
            self.assertEqual(encoded.shape[-2:], (13, 13), msg=f"decoder scale D{scale}")
            self.assertEqual(decoder(decoder_input).shape, (1, 4, 2))

    def test_all_scales_have_exact_zero_initial_output(self) -> None:
        torch.manual_seed(11)
        _, decoders = build_multiscale_modules()
        with torch.no_grad():
            for scale, spec in SCALE_SPECS.items():
                decoder_input = torch.randn(1, spec.decoder_input_channels, 8, 8)
                output = decoders[str(scale)](decoder_input)
                self.assertEqual(output.shape, (1, 4, 2))
                self.assertTrue(torch.equal(output, torch.zeros_like(output)))


class ZeroInitializationGradientTests(unittest.TestCase):
    def test_fc2_learns_first_then_upstream_receives_gradient(self) -> None:
        torch.manual_seed(23)
        spec = SCALE_SPECS[8]
        decoder = ResidualGeometryDecoder(
            spec.decoder_input_channels,
            spec.down_blocks,
            spec.max_delta_px,
        )
        optimizer = torch.optim.SGD(decoder.parameters(), lr=1e-5)
        decoder_input = torch.randn(2, spec.decoder_input_channels, 16, 16)

        optimizer.zero_grad(set_to_none=True)
        first_output = decoder(decoder_input)
        first_output.sum().backward()

        self.assertGreater(float(decoder.fc2.weight.grad.norm()), 0.0)
        self.assertGreater(float(decoder.fc2.bias.grad.norm()), 0.0)
        self.assertEqual(float(decoder.fc1.weight.grad.abs().max()), 0.0)
        self.assertEqual(float(decoder.stem_conv.weight.grad.abs().max()), 0.0)
        optimizer.step()
        self.assertGreater(float(decoder.fc2.weight.detach().norm()), 0.0)

        optimizer.zero_grad(set_to_none=True)
        second_output = decoder(decoder_input)
        second_output.sum().backward()

        for name, gradient in (
            ("fc1", decoder.fc1.weight.grad),
            ("last_downblock", decoder.down_blocks[-1].conv2.weight.grad),
            ("stem", decoder.stem_conv.weight.grad),
        ):
            self.assertIsNotNone(gradient, msg=name)
            self.assertTrue(bool(torch.isfinite(gradient).all()), msg=name)
            self.assertGreater(float(gradient.norm()), 0.0, msg=name)


if __name__ == "__main__":
    unittest.main()
