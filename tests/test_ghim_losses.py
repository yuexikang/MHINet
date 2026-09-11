import unittest
import torch
from mhinet.engine.ghim_losses import ghim_loss
from mhinet.engine.train import TrainConfig


class GHIMLossTests(unittest.TestCase):
    def inputs(self):
        warp = torch.tensor([[[[-.4, .6], [-.4, .6]], [[-.5, -.5], [.5, .5]]]], requires_grad=True)
        matrix = torch.eye(3)[None].clone().requires_grad_()
        return {'coarse_warp': warp,
                'coarse_matchability': torch.full((1, 1, 2, 2), .7, requires_grad=True),
                'match_probabilities': torch.full((1, 4, 5), .2, requires_grad=True),
                'H_A_to_B_norm': matrix, 'fit_succeeded': torch.tensor([True])}

    def test_loss_and_gradients(self):
        outputs = self.inputs()
        loss = ghim_loss(outputs, torch.eye(3)[None], torch.ones(1, 1, 4, 4))
        self.assertAlmostEqual(loss['H'].item(), 0)
        self.assertGreater(loss['geo'].item(), 0)
        loss['total'].backward()
        for key in ('coarse_warp', 'coarse_matchability', 'match_probabilities'):
            self.assertTrue(torch.isfinite(outputs[key].grad).all())
            self.assertGreater(outputs[key].grad.abs().sum().item(), 0)

    def test_failed_singular_H_is_isolated_but_coarse_trains(self):
        outputs = self.inputs()
        outputs['H_A_to_B_norm'] = torch.zeros(1, 3, 3, requires_grad=True)
        outputs['fit_succeeded'] = torch.tensor([False])
        loss = ghim_loss(outputs, torch.eye(3)[None], torch.zeros(1, 1, 4, 4))
        self.assertEqual(loss['valid_H_pairs'].item(), 0)
        self.assertEqual(loss['H'].item(), 0)
        self.assertEqual(loss['geo'].item(), 0)
        loss['total'].backward()
        self.assertTrue(torch.isfinite(outputs['H_A_to_B_norm'].grad).all())
        self.assertGreater(outputs['match_probabilities'].grad.abs().sum().item(), 0)

    def test_new_configuration(self):
        cfg = TrainConfig.from_json('configs/train_frozen_dino_mvt.json')
        self.assertEqual(cfg.profile, 'frozen_dino_mvt')

