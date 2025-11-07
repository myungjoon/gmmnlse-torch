import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class BigBlock(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(), device=device
        )
    def forward(self, x):
        return self.net(x)

big_block = BigBlock(device='cuda')

def block_fn(x):
    # checkpoint로 감싸기 쉬운 형태: 입력/출력은 텐서(들), 내부는 순수 연산
    return big_block(x)

x = torch.randn(32, 4096, requires_grad=True).cuda()
y = checkpoint(block_fn, x)    # <-- 여기서 메모리 절약
loss = y.pow(2).mean()
loss.backward()