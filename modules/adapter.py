#       Input (D_model)
#          |
#          |  (原始子层，如FFN或Attention)
#          |
#       Output_sublayer (D_model)
#       /      \
#      |        | (Adapter Module)
#      |        |
#      |    Linear (D_model -> Bottleneck_dim)
#      |        |
#      |      ReLU/GeLU
#      |        |
#      |    Linear (Bottleneck_dim -> D_model)
#      |        |
#      |      Dropout (optional)
#      |        |
#       \      / (Add)
#        \    /
#       Output_sublayer + Output_adapter

import torch
import torch.nn as nn
import math

# 0. 辅助模块：一个简单的前馈网络
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU() # 或者 nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# 1. Adapter 模块定义
class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck_dim, dropout=0.1):
        super().__init__()
        self.down_project = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.ReLU() # 或者 nn.GELU()
        self.up_project = nn.Linear(bottleneck_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        # 初始化 up_project 的权重为接近零，使得初始时适配器接近恒等变换
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)


    def forward(self, x):
        # x 是前一个子层 (如FFN或Attention) 的输出
        down_projected = self.down_project(x)
        activated = self.activation(down_projected)
        up_projected = self.up_project(activated)
        up_projected = self.dropout(up_projected)
        # 通常，适配器的输出会加到原始子层的输出上
        # 在这里，我们设计适配器模块本身返回其效果，外部进行相加
        return up_projected # 返回适配器计算出的增量

# 2. 带 Adapter 的 Transformer Encoder Layer (简化版)
class TransformerEncoderLayerWithAdapter(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, adapter_bottleneck_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 为 FFN 子层添加一个 Adapter
        # 也可以为 Attention 子层添加，这里仅演示 FFN 后添加
        self.ffn_adapter = Adapter(d_model, adapter_bottleneck_dim, dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # --- Self-Attention Block ---
        src_norm1 = self.norm1(src)
        attn_output, _ = self.self_attn(src_norm1, src_norm1, src_norm1,
                                        attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output) # Residual connection for attention

        # --- Feed-Forward Block with Adapter ---
        src_norm2 = self.norm2(src)
        ffn_output = self.ffn(src_norm2)

        # 应用 Adapter
        # 方式1: 适配器作用于 FFN 的输出，然后加到 FFN 的输出上，再进行主残差连接
        # adapter_effect = self.ffn_adapter(ffn_output)
        # src = src + self.dropout2(ffn_output + adapter_effect)

        # 方式2: 适配器作用于 FFN 的输出，其结果与 FFN 输出一起加到 FFN 的输入上 (Houlsby et al. 2019 风格)
        # This means the adapter output is added to the main residual stream
        adapter_output = self.ffn_adapter(ffn_output) # Adapter operates on FFN output
        # Add FFN output and Adapter output to the input of the FFN block (src)
        src = src + self.dropout2(ffn_output) + adapter_output # Additive adapter

        return src

# 3. 演示 Adapter Tuning 的参数冻结和训练设置
def demo_adapter_tuning():
    D_MODEL = 256
    N_HEAD = 4
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1
    ADAPTER_BOTTLENECK_DIM = 64 # 适配器瓶颈维度，远小于 D_MODEL
    VOCAB_SIZE = 1000
    SEQ_LEN = 20
    BATCH_SIZE = 8

    # 假设我们有一个包含多个 EncoderLayer 的完整模型
    # 这里仅用一个 Layer 来演示
    base_model_layer = TransformerEncoderLayerWithAdapter(
        D_MODEL, N_HEAD, DIM_FEEDFORWARD, DROPOUT, ADAPTER_BOTTLENECK_DIM
    )

    # --- 参数冻结 ---
    # 冻结所有非 Adapter 参数
    for name, param in base_model_layer.named_parameters():
        if 'adapter' not in name: # 关键：只训练名字中包含 'adapter' 的参数
            param.requires_grad = False
        else:
            param.requires_grad = True
            print(f"Training parameter: {name} with shape {param.shape}")

    # 验证哪些参数是可训练的
    trainable_params = sum(p.numel() for p in base_model_layer.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in base_model_layer.parameters())
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters (Adapters): {trainable_params}")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    # --- 模拟训练 ---
    # 假设有一个分类头
    classifier_head = nn.Linear(D_MODEL, 10) # 10个类别
    
    # 将分类头的参数也设为可训练
    for param in classifier_head.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, list(base_model_layer.parameters()) + list(classifier_head.parameters())),
        lr=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    # 模拟输入数据
    dummy_input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    dummy_embeddings = nn.Embedding(VOCAB_SIZE, D_MODEL)(dummy_input_ids) # (B, S, D_MODEL)
    dummy_labels = torch.randint(0, 10, (BATCH_SIZE,))

    # 训练步骤
    base_model_layer.train()
    classifier_head.train()

    optimizer.zero_grad()
    output_hidden_states = base_model_layer(dummy_embeddings) # (B, S, D_MODEL)
    # 通常取 [CLS] token 或平均池化作为句子表示
    sentence_representation = output_hidden_states[:, 0, :] # (B, D_MODEL)
    logits = classifier_head(sentence_representation) # (B, num_classes)
    loss = criterion(logits, dummy_labels)
    loss.backward()
    optimizer.step()

    print(f"\nDummy training step completed. Loss: {loss.item()}")

if __name__ == '__main__':
    print("--- Adapter Tuning Demo ---")
    demo_adapter_tuning()