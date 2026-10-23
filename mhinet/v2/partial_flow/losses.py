"""Separately normalized positive/negative BCE, no coordinate objective."""
import torch.nn.functional as F
from .labels import MATCH, NO_MATCH


def existence_loss(logits, state):
    positive, negative = state == MATCH, state == NO_MATCH
    zero = logits.sum()*0
    p = F.softplus(logits[positive]).mean() if positive.any() else zero
    n = F.softplus(-logits[negative]).mean() if negative.any() else zero
    return p+n
