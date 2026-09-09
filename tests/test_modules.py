"""Unit tests for the parameterized MHINet v1 refinement modules."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from mhinet.modules import (
    EXPECTED_MAINLINE_TRAINABLE_NEW_PARAMETERS,
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
        self.assertEqual(sum(parameter.numel() for parameter in block.parameters()), 37_056)
        output = block(torch.randn(2, 64, 13, 17))
        self.assertEqual(output.shape, (2, 64, 7, 9))

    def test_factory_shapes_initialization_and_parameter_count(self) -> None:
        adapters, decoders = build_multiscale_modules()
        self.assertEqual(tuple(adapters.keys()), ("8", "4", "2", "1"))
        self.assertEqual(tuple(decoders.keys()), ("8", "4", "2", "1"))
        self.assertEqual(count_new_parameters(adapters, decoders), EXPECTED_NEW_PARAMETERS)
        self.assertEqual(
            count_new_parameters(
                *(adapters[str(scale)] for scale in (8, 4, 2)),
                *(decoders[str(scale)] for scale in (8, 4, 2)),
            ),
            EXPECTED_MAINLINE_TRAINABLE_NEW_PARAMETERS,
        )

        expected_counts = {
            8: (16_384, 227_714),
            4: (16_384, 264_770),
            2: (8_192, 299_778),
            1: (8_192, 335_298),
        }
        for scale, spec in SCALE_SPECS.items():
            adapter = adapters[str(scale)]
            decoder = decoders[str(scale)]
            self.assertEqual(
                spec.decoder_input_channels,
                spec.correlation_candidates,
            )
            self.assertEqual(adapter.out_channels, spec.adapter_channels)
            self.assertEqual(decoder.input_channels, spec.decoder_input_channels)
            self.assertEqual(decoder.input_spatial_size, spec.spatial_size)
            self.assertEqual(len(decoder.layers), spec.down_blocks)
            self.assertEqual(decoder.max_delta_px, spec.max_delta_px)
            self.assertEqual(
                sum(parameter.numel() for parameter in adapter.parameters()),
                expected_counts[scale][0],
            )
            self.assertEqual(
                sum(parameter.numel() for parameter in decoder.parameters()),
                expected_counts[scale][1],
            )
            self.assertTrue(
                torch.equal(
                    decoder.out_conv.weight, torch.zeros_like(decoder.out_conv.weight)
                )
            )
            self.assertTrue(
                torch.equal(
                    decoder.out_conv.bias, torch.zeros_like(decoder.out_conv.bias)
                )
            )

            for module in decoder.modules():
                if isinstance(module, nn.GroupNorm):
                    self.assertTrue(torch.equal(module.weight, torch.ones_like(module.weight)))
                    self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))
                if isinstance(module, nn.Conv2d) and module is not decoder.out_conv:
                    self.assertGreater(int(torch.count_nonzero(module.weight)), 0)
                    if module.bias is not None:
                        self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))
            self.assertTrue(decoder.in_conv.bias is not None)
            for block in decoder.layers:
                self.assertEqual(block.conv.kernel_size, (3, 3))
                self.assertEqual(block.conv.stride, (1, 1))
                self.assertEqual(block.conv.padding, (1, 1))
                self.assertIsNotNone(block.conv.bias)
                self.assertEqual(block.norm.num_groups, 8)
                self.assertIsInstance(block.activation, nn.ReLU)
                self.assertEqual(block.pool.kernel_size, 2)
                self.assertEqual(block.pool.stride, 2)
                self.assertTrue(block.pool.ceil_mode)

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
                input_spatial_size=spatial,
            ).to(device="meta")
            decoder_input = torch.empty(
                1,
                spec.decoder_input_channels,
                spatial,
                spatial,
                device="meta",
            )
            encoded = decoder.in_conv(decoder_input)
            for block in decoder.layers:
                encoded = block(encoded)
            self.assertEqual(encoded.shape[-2:], (2, 2), msg=f"decoder scale D{scale}")
            self.assertEqual(decoder(decoder_input).shape, (1, 4, 2))

    def test_all_scales_have_exact_zero_initial_output(self) -> None:
        torch.manual_seed(11)
        with torch.no_grad():
            for scale, spec in SCALE_SPECS.items():
                decoder = ResidualGeometryDecoder(
                    spec.decoder_input_channels,
                    2,
                    spec.max_delta_px,
                    input_spatial_size=8,
                )
                decoder_input = torch.randn(1, spec.decoder_input_channels, 8, 8)
                output = decoder(decoder_input)
                self.assertEqual(output.shape, (1, 4, 2))
                self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_output_grid_maps_row_major_to_tl_tr_bl_br(self) -> None:
        decoder = ResidualGeometryDecoder(2, 0, 1.0)
        with torch.no_grad():
            decoder.in_conv.weight.zero_()
            decoder.in_conv.bias.zero_()
            decoder.in_conv.weight[0, 0, 0, 0] = 1.0
            decoder.in_conv.weight[1, 1, 0, 0] = 1.0
            decoder.out_conv.weight.zero_()
            decoder.out_conv.bias.zero_()
            decoder.out_conv.weight[0, 0, 0, 0] = 1.0
            decoder.out_conv.weight[1, 1, 0, 0] = 1.0
        decoder_input = torch.tensor(
            [[
                [[0.1, 0.2], [0.3, 0.4]],
                [[-0.1, -0.2], [-0.3, -0.4]],
            ]]
        )
        expected = torch.tensor(
            [[[0.1, -0.1], [0.2, -0.2], [0.3, -0.3], [0.4, -0.4]]]
        ).tanh()
        torch.testing.assert_close(decoder(decoder_input), expected)


class ZeroInitializationGradientTests(unittest.TestCase):
    def test_out_conv_learns_first_then_upstream_receives_gradient(self) -> None:
        torch.manual_seed(23)
        spec = SCALE_SPECS[8]
        decoder = ResidualGeometryDecoder(
            spec.decoder_input_channels,
            3,
            spec.max_delta_px,
            input_spatial_size=16,
        )
        optimizer = torch.optim.SGD(decoder.parameters(), lr=1e-5)
        decoder_input = torch.randn(2, spec.decoder_input_channels, 16, 16)

        optimizer.zero_grad(set_to_none=True)
        first_output = decoder(decoder_input)
        first_output.sum().backward()

        self.assertGreater(float(decoder.out_conv.weight.grad.norm()), 0.0)
        self.assertGreater(float(decoder.out_conv.bias.grad.norm()), 0.0)
        self.assertEqual(float(decoder.layers[-1].conv.weight.grad.abs().max()), 0.0)
        self.assertEqual(float(decoder.in_conv.weight.grad.abs().max()), 0.0)
        optimizer.step()
        self.assertGreater(float(decoder.out_conv.weight.detach().norm()), 0.0)

        optimizer.zero_grad(set_to_none=True)
        second_output = decoder(decoder_input)
        second_output.sum().backward()

        for name, gradient in (
            ("last_downblock", decoder.layers[-1].conv.weight.grad),
            ("stem", decoder.in_conv.weight.grad),
        ):
            self.assertIsNotNone(gradient, msg=name)
            self.assertTrue(bool(torch.isfinite(gradient).all()), msg=name)
            self.assertGreater(float(gradient.norm()), 0.0, msg=name)


if __name__ == "__main__":
    unittest.main()
