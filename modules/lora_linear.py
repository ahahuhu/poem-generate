import select
import torch
from torch import nn

class LoRALinear(nn.Module):

    def __init__(self, in_features, out_features, r: int = 4, bias=True, device=None, dtype=None):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features,
                                bias=bias,
                                device=device,
                                dtype=dtype)
        self.A = nn.Parameter(torch.randn((in_features, r)), requires_grad=True)
        self.B = nn.Parameter(torch.zeros((r, out_features)), requires_grad=True)
        for parameter in self.linear.parameters():
            parameter.requires_grad = False
    
    def forward(self, data: torch.Tensor):
        """前向传播

        Args:
            data (torch.Tensor): (bs, sl, in_feature)

        Returns:
            _type_: 线性变化的结果 (bs, sl, out_feature)
        """
        return self.linear(data) + data@self.A@self.B

## TODO 进行测试


if __name__ == "__main__":
    layer = LoRALinear(768, 512, r=8)
    x = torch.randn(16, 768)  # batch=16, input_dim=768
    y = layer(x)
    print(y.shape)  # 应输出 torch.Size([16, 512])