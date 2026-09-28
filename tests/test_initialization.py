import unittest
import torch
from torch import nn
from mhinet.models.initialization import initialize_ablation


class Fake(nn.Module):
    def __init__(self):
        super().__init__()
        self.parts=nn.ModuleDict({k:nn.Linear(3,3) for k in ['dino','vgg','mvt','dedode','stage1_head_parameters','new_modules']})
        self.refinement_decoders=nn.ModuleDict()
    def _all_parameter_groups(self):return {k:list(v.parameters()) for k,v in self.parts.items()}


class InitializationTests(unittest.TestCase):
    def test_preserved_and_reset(self):
        model=Fake();state=torch.get_rng_state().clone()
        r=initialize_ablation(model,'random_except_dino_vgg',0)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        for k in ['dino','vgg','new_modules']:self.assertEqual(r['before'][k],r['after'][k])
        for k in ['mvt','dedode','stage1_head_parameters']:self.assertNotEqual(r['before'][k],r['after'][k])
    def test_pretrained_unchanged(self):
        r=initialize_ablation(Fake(),'pretrained',0)
        self.assertEqual(r['before'],r['after'])
