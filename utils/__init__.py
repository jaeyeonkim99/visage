import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    cnt = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        cnt += nn

    return cnt


def count_trainable_parameters(model: nn.Module) -> int:
    cnt = 0
    for p in list(model.parameters()):
        if p.requires_grad:
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            cnt += nn

    return cnt
