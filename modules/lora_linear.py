import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r: int = 4, alpha: float = 1.0, bias=True, device=None, dtype=None, use_lora: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.lora_A = nn.Parameter(torch.randn(r, in_features, device=device, dtype=dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=dtype))
        self.scaling = alpha / r
        self.use_lora = use_lora
        if use_lora:
            self.linear.weight.requires_grad_(False)
            if self.linear.bias is not None:
                self.linear.bias.requires_grad_(False)

    def forward(self, data: torch.Tensor):
        out = self.linear(data)
        if self.use_lora:
            # data: (batch, seq_len, in_features)
            # lora_A: (r, in_features)
            # lora_B: (out_features, r)
            # 先做 (batch, seq_len, in_features) @ (in_features, r) -> (batch, seq_len, r)
            lora_A_proj = torch.matmul(data, self.lora_A.T)
            # 再做 (batch, seq_len, r) @ (r, out_features) -> (batch, seq_len, out_features)
            lora_out = torch.matmul(lora_A_proj, self.lora_B.T) * self.scaling
            # lora_out shape: (batch, seq_len, out_features)
            # out shape: (batch, seq_len, out_features)
            
            # Debug information to understand shape mismatch
            if out.shape != lora_out.shape:
                print(f"Shape mismatch: out {out.shape}, lora_out {lora_out.shape}")
                print(f"linear: out {self.linear.weight.shape}, lora_out {lora_out.shape}")
                print(f"LoRA matrices: lora_A {self.lora_A.shape}, lora_B {self.lora_B.shape}")
                print(f"Input data shape: {data.shape}")
            
            out = out + lora_out
        return out

if __name__ == "__main__":
    layer = LoRALinear(768, 512, r=8)
    x = torch.randn(16, 768)  # batch=16, input_dim=768
    y = layer(x)
    print(y.shape)  # 应输出 torch.Size([16, 512])