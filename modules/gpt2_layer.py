from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    output = dense_layer(output)
    output = dropout(output)
    output = input + output
    
    return output

  def forward(self, hidden_states, attention_mask):
    l1_out = self.attention_layer_norm(hidden_states)


    output_attention = self.self_attention(l1_out, attention_mask)
    output_res = self.add(hidden_states, output_attention, self.attention_dense, self.attention_dropout)
    l2_out = self.out_layer_norm(output_res)

    # ffn
    ffn_output = self.interm_af(self.interm_dense(l2_out))
    output = self.add(output_res, ffn_output, self.out_dense, self.out_dropout)
    
    return output
