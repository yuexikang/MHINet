"""Optional common pretrained MVT/MatchEmbedding initialization for comparisons."""
import torch


def initialize_locator(system, config):
    path = config.get('frozen_locator_checkpoint')
    if not path:
        return dict(pretrained_modules=['DINOv3'],
                    fresh_modules=['MVT','VGG','CGMDP','Flow Head','D8/D2 temperatures','QRRU'])
    load_locator_weights(system.shared, path)
    system.shared.set_training_groups(mvt=False, vgg=True, dedode=True, ghim_head=False)
    return dict(pretrained_modules=['DINOv3','MVT','MatchEmbedding locator'],
                fresh_modules=['VGG','CGMDP','D8/D2 temperatures','QRRU'],
                frozen_modules=['DINOv3','MVT','MatchEmbedding locator'],
                locator_checkpoint=str(path),
                locator_scope='Decoder, positional buffers, no-match embedding and logit; identical source for v1/v2')


def load_locator_weights(shared, path):
    """Load MVT and the full original Flow head from a local checkpoint."""
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if 'mvt' in payload and 'head' in payload:
        shared.mvt.load_state_dict(payload['mvt'], strict=True)
        shared.stage1_head.load_state_dict(payload['head'], strict=True)
    else:
        state = payload['model']
        for prefix, module in (
            ('shared.descriptor.encoder.frozen_dinov3.multi_view_transformer.', shared.mvt),
            ('shared.stage1_head.', shared.stage1_head),
        ):
            module.load_state_dict({k[len(prefix):]: v for k, v in state.items()
                                    if k.startswith(prefix)}, strict=True)
