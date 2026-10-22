"""Optional common pretrained MVT/MatchEmbedding initialization for comparisons."""
import torch


def initialize_locator(system, config):
    path = config.get('frozen_locator_checkpoint')
    if not path:
        return dict(pretrained_modules=['DINOv3'],
                    fresh_modules=['MVT','VGG','CGMDP','Flow Head','D8/D2 temperatures','QRRU'])
    # Local, previously produced MHINet checkpoint. Only these two subtrees load.
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    state = payload['model']
    modules = {
        'shared.descriptor.encoder.frozen_dinov3.multi_view_transformer.': system.shared.mvt,
        'shared.stage1_head.': system.shared.stage1_head,
    }
    for prefix, module in modules.items():
        module.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}, strict=True)
    system.shared.set_training_groups(mvt=False, vgg=True, dedode=True, ghim_head=False)
    return dict(pretrained_modules=['DINOv3','MVT','MatchEmbedding locator'],
                fresh_modules=['VGG','CGMDP','D8/D2 temperatures','QRRU'],
                frozen_modules=['DINOv3','MVT','MatchEmbedding locator'],
                locator_checkpoint=str(path),
                locator_scope='Decoder, positional buffers, no-match embedding and logit; identical source for v1/v2')
