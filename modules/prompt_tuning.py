import torch
import torch.nn as nn

# 1. 软提示模块 (本质上是一个可训练的 Embedding 层)
class SoftPrompt(nn.Module):
    def __init__(self, num_prompt_tokens, d_model):
        super().__init__()
        self.num_prompt_tokens = num_prompt_tokens
        self.d_model = d_model
        # 软提示参数，初始化可以采用多种策略，例如从词汇表中的某些词嵌入初始化
        self.prompt_embeddings = nn.Embedding(num_prompt_tokens, d_model)
        # 或者直接用 Parameter
        # self.prompt_vectors = nn.Parameter(torch.randn(num_prompt_tokens, d_model))

    def forward(self, batch_size):
        # 返回 (batch_size, num_prompt_tokens, d_model) 的提示嵌入
        # 如果用 nn.Embedding, 需要一个 dummy input indices
        prompt_indices = torch.arange(self.num_prompt_tokens, device=self.prompt_embeddings.weight.device)
        prompts = self.prompt_embeddings(prompt_indices) # (num_prompt_tokens, d_model)
        return prompts.unsqueeze(0).repeat(batch_size, 1, 1) # (batch_size, num_prompt_tokens, d_model)
        # 如果用 nn.Parameter:
        # return self.prompt_vectors.unsqueeze(0).repeat(batch_size, 1, 1)

# 2. 带 Prompt Tuning 的 Transformer 模型 (简化版)
class TransformerModelWithPromptTuning(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, num_prompt_tokens):
        super().__init__()
        self.d_model = d_model
        self.num_prompt_tokens = num_prompt_tokens

        # 预训练模型部分 (这里用 PyTorch 内置的简化)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        # 通常这里会加载一个预训练好的 Transformer

        # 软提示模块
        self.soft_prompt = SoftPrompt(num_prompt_tokens, d_model)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (batch_size, seq_len)
        # attention_mask: (batch_size, seq_len) for padding, if any

        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

        # 1. 获取原始词嵌入
        token_embeds = self.token_embedding(input_ids) # (batch_size, seq_len, d_model)

        # 2. 获取软提示嵌入
        prompt_embeds = self.soft_prompt(batch_size) # (batch_size, num_prompt_tokens, d_model)

        # 3. 拼接软提示和词嵌入
        # [p1, p2, ..., pk, w1, w2, ..., wn]
        combined_embeds = torch.cat([prompt_embeds, token_embeds], dim=1) # (B, num_prompt + seq_len, D_MODEL)

        # 4. 调整 Attention Mask (如果原始输入有padding)
        # 软提示部分总是被注意到 (mask=0 or True depending on mask type)
        # 真实token部分使用原始mask
        if attention_mask is not None:
            # Assuming attention_mask is 0 for attended, 1 for not attended (like Hugging Face)
            # Or False for attended, True for not attended (like PyTorch nn.MultiheadAttention key_padding_mask)
            # Let's assume key_padding_mask (True means ignore)
            prompt_mask = torch.zeros(batch_size, self.num_prompt_tokens, dtype=torch.bool, device=input_ids.device)
            # If original attention_mask was (B, S), and 1 means padding.
            # We need a combined mask (B, num_prompt + S)
            # For nn.TransformerEncoderLayer, src_key_padding_mask should be (N, S)
            # where S is source sequence length. If a BoolTensor is provided,
            # True values are positions that ARE NOT ATTENDED.
            if attention_mask.dtype == torch.float: # Often 1.0 for attended, 0.0 for padded
                 # Convert to boolean mask: True for padding
                input_key_padding_mask = (attention_mask == 0)
            else: # Assuming boolean: True for padding
                input_key_padding_mask = attention_mask

            combined_key_padding_mask = torch.cat([prompt_mask, input_key_padding_mask], dim=1)
        else:
            combined_key_padding_mask = None


        # 5. 通过 Transformer Encoder
        # 注意：TransformerEncoderLayer 内部的 MHA 需要 (L, N, E) 或 (N, L, E) if batch_first=True
        # 这里 L 是序列长度，N 是 batch size，E 是特征维度
        # combined_embeds is (B, num_prompt + seq_len, D_MODEL) which is correct for batch_first=True
        encoder_output = self.transformer_encoder(combined_embeds, src_key_padding_mask=combined_key_padding_mask)
        # Output: (B, num_prompt + seq_len, D_MODEL)

        # 通常，我们只关心原始token部分的输出，或者特定位置的输出
        # 例如，用于分类任务时，可以取第一个真实token（如果前面有CLS）或最后一个token的输出，
        # 或者对真实token的输出进行平均池化。
        # 这里返回所有真实token的输出
        output_tokens_only = encoder_output[:, self.num_prompt_tokens:, :] # (B, seq_len, D_MODEL)
        return output_tokens_only


# 3. 演示 Prompt Tuning 的参数冻结和训练设置
def demo_prompt_tuning():
    D_MODEL = 256
    N_HEAD = 4
    NUM_ENCODER_LAYERS = 2
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1
    NUM_PROMPT_TOKENS = 10 # 软提示的长度
    VOCAB_SIZE = 1000
    SEQ_LEN = 20
    BATCH_SIZE = 8

    model = TransformerModelWithPromptTuning(
        VOCAB_SIZE, D_MODEL, N_HEAD, NUM_ENCODER_LAYERS, DIM_FEEDFORWARD, DROPOUT, NUM_PROMPT_TOKENS
    )

    # --- 参数冻结 ---
    # 冻结所有非 SoftPrompt 参数
    for name, param in model.named_parameters():
        if 'soft_prompt' not in name: # 关键：只训练 'soft_prompt' 模块的参数
            param.requires_grad = False
        else:
            param.requires_grad = True
            print(f"Training parameter: {name} with shape {param.shape}")

    # 验证哪些参数是可训练的
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters (Soft Prompts): {trainable_params}")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.4f}%") # 注意是4位小数，因为比例可能很小

    # --- 模拟训练 ---
    classifier_head = nn.Linear(D_MODEL, 10) # 10个类别 (通常这个也需要训练)
    for param in classifier_head.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, list(model.parameters()) + list(classifier_head.parameters())),
        lr=1e-3 # Prompt tuning 可能需要稍大的学习率
    )
    criterion = nn.CrossEntropyLoss()

    # 模拟输入数据
    dummy_input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    # 模拟 padding mask (0 表示 padding, 1 表示真实 token) -> PyTorch MHA needs True for padding
    dummy_attention_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool) # No padding in this example
    # If we had padding, e.g. last 5 tokens are padding for first sample:
    # dummy_attention_mask[0, -5:] = False # False means attend, True means pad
    # Let's make it compatible with src_key_padding_mask (True means pad)
    # So, if all ones means all are attended, then we need to make it False
    dummy_src_key_padding_mask = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.bool) # False means attend

    dummy_labels = torch.randint(0, 10, (BATCH_SIZE,))

    # 训练步骤
    model.train()
    classifier_head.train()

    optimizer.zero_grad()
    output_hidden_states = model(dummy_input_ids, attention_mask=dummy_src_key_padding_mask) # (B, S, D_MODEL)
    sentence_representation = output_hidden_states[:, 0, :] # 取第一个真实token的表示
    logits = classifier_head(sentence_representation)
    loss = criterion(logits, dummy_labels)
    loss.backward()
    optimizer.step()

    print(f"\nDummy training step completed. Loss: {loss.item()}")

if __name__ == '__main__':
    print("\n--- Prompt Tuning Demo ---")
    demo_prompt_tuning()