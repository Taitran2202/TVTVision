import torch.nn as nn
from .safecountblock import SAFECountBlock


class SAFECountMultiBlock(nn.Module):
    def __init__(self, num_block, embed_dim, mid_dim, head, dropout):
        super(SAFECountMultiBlock, self).__init__()
        self.blocks = nn.ModuleList([SAFECountBlock(
            embed_dim=embed_dim,
            mid_dim=mid_dim,
            head=head,
            dropout=dropout,
        ) for _ in range(num_block)])

    def forward(self, tgt, src):
        for block in self.blocks:
            fq = block(tgt, src)

        return fq
