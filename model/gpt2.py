import torch
from torch import nn
from transformers import GPT2LMHeadModel, AutoModel

from config import GPT2Config
from model.base_gpt import GPTPreTrainedModel
from modules.gpt2_layer import GPT2Layer
from modules.lora_linear import LoRALinear
import os

def get_extended_attention_mask(attention_mask: torch.Tensor, dtype) -> torch.Tensor:
    # attention_mask [batch_size, seq_length]
    assert attention_mask.dim() == 2
    # [batch_size, 1, 1, seq_length] for multi-head attention
    extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask

class GPT2Model(GPTPreTrainedModel):
  
    def __init__(self, config):
        super().__init__(config)
        self.config = config
    
        # Embedding layers.
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
    
        # Register position_ids (1, len position emb) to buffer because it is a constant.
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)
    
        # GPT-2 layers.
        self.gpt_layers = nn.ModuleList([GPT2Layer(config) for _ in range(config.num_hidden_layers)])
    
        # [CLS] token transformations.
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()
    
        # Final layer norm.
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
        self.init_weights()

    def embed(self, input_ids):
        input_shape = input_ids.size() # shape (b, t)
        seq_length = input_shape[1]
    
        inputs_embeds = None
    
        # 进行token_id的向量化，进行positional的向量化
        inputs_embeds = self.word_embedding(input_ids) #(b,t,d)

        pos_ids = self.position_ids[:, :seq_length]
        pos_embeds = None

        pos_embeds = self.pos_embedding(pos_ids) #(1, t, d)
        return self.embed_dropout(inputs_embeds + pos_embeds)  #(b, t, d)


    def encode(self, hidden_states, attention_mask):
        """
        hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        """
        # Get the extended attention mask for self-attention.
        # Returns extended_attention_mask of size [batch_size, 1, 1, seq_len].
        # Distinguishes between non-padding tokens (with a value of 0) and padding tokens
        # (with a value of a large negative number).
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

        # Pass the hidden states through the encoder layers.
        for i, layer_module in enumerate(self.gpt_layers):
          # Feed the encoding from the last bert_layer to the next.
          hidden_states = layer_module(hidden_states, extended_attention_mask)

        return hidden_states

    def forward(self, input_ids, attention_mask):
        """
        input_ids: [batch_size, seq_len], seq_len is the max length of the batch
        attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
        """
        # Get the embedding for each input token.
        embedding_output = self.embed(input_ids=input_ids)

        # Feed to a transformer (a stack of GPTLayers).
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)
        sequence_output = self.final_layer_norm(sequence_output)

        # Get the hidden state of the final token.
        last_non_pad_idx = attention_mask.sum(dim=1) - 1  # Subtract 1 to get last index
        last_token = sequence_output[torch.arange(sequence_output.shape[0]), last_non_pad_idx]

        return {'last_hidden_state': sequence_output, 'last_token': last_token}

    def hidden_state_to_token(self, hidden_state):
        """
        GPT-2 uses weight tying with the input word embeddings. The logits are the dot product between output hidden states
        and the word embedding weights:
    
          return hidden_state(s) * E^T
        """
        ### YOUR CODE HERE
        raise NotImplementedError


    @classmethod
    def from_pretrained(cls, model_name='uer/gpt2-chinese-cluecorpussmall', model_dir:str = 'cache/pretrained_model', d=768, l=12, num_heads=12, use_lora: bool = False):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            AutoModel.from_pretrained(model_name).save_pretrained(model_dir)
        gpt_model = GPT2LMHeadModel.from_pretrained(model_dir).eval()
        our_model = GPT2Model(GPT2Config(hidden_size=d, num_hidden_layers=l,num_attention_heads=num_heads,
                                         intermediate_size=d*3, use_lora = use_lora)).eval()
        # for name, param in gpt_model.named_parameters():
        #   print(name)

        # Load word and positional embeddings.
        our_model.word_embedding.load_state_dict(gpt_model.transformer.wte.state_dict())
        our_model.pos_embedding.load_state_dict(gpt_model.transformer.wpe.state_dict())
        for i in range(l):
            l = our_model.gpt_layers[i]
            # Remap the Q,K,V weights from a conv1d to 3 linear projections
            l.self_attention.query.weight.data = gpt_model.state_dict()[f'transformer.h.{i}.attn.c_attn.weight'][:, :d].T
            l.self_attention.query.bias.data = gpt_model.state_dict()[f'transformer.h.{i}.attn.c_attn.bias'][:d]
            l.self_attention.key.weight.data = gpt_model.state_dict()[f'transformer.h.{i}.attn.c_attn.weight'][:, d:d*2].T
            l.self_attention.key.bias.data = gpt_model.state_dict()[f'transformer.h.{i}.attn.c_attn.bias'][d:d*2]
            l.self_attention.value.weight.data = gpt_model.state_dict()[f'transformer.h.{i}.attn.c_attn.weight'][:, d*2:].T
            l.self_attention.value.bias.data = gpt_model.state_dict()[f'transformer.h.{i}.attn.c_attn.bias'][d*2:]
    
            # Remap final dense layer in MHA.
            l.attention_dense.weight.data = gpt_model.state_dict()[f'transformer.h.{i}.attn.c_proj.weight'].T
            l.attention_dense.bias.data = gpt_model.state_dict()[f'transformer.h.{i}.attn.c_proj.bias']
    
            # Remap attention layer norm.
            l.attention_layer_norm.weight.data = gpt_model.state_dict()[f'transformer.h.{i}.ln_1.weight']
            l.attention_layer_norm.bias.data = gpt_model.state_dict()[f'transformer.h.{i}.ln_1.bias']
    
            # Remap post-attention MLP layers.
            # print("++++++++++++++++++++")
            # print(l.interm_dense.linear.weight.shape)
            # print(gpt_model.state_dict()[f'transformer.h.{i}.mlp.c_fc.weight'].shape)
            l.interm_dense.weight.data = gpt_model.state_dict()[f'transformer.h.{i}.mlp.c_fc.weight'].T
            l.interm_dense.bias.data = gpt_model.state_dict()[f'transformer.h.{i}.mlp.c_fc.bias']
            l.out_dense.weight.data = gpt_model.state_dict()[f'transformer.h.{i}.mlp.c_proj.weight'].T
            l.out_dense.bias.data = gpt_model.state_dict()[f'transformer.h.{i}.mlp.c_proj.bias']
    
            # Remap second layer norm weights.
            l.out_layer_norm.weight.data = gpt_model.state_dict()[f'transformer.h.{i}.ln_2.weight']
            l.out_layer_norm.bias.data = gpt_model.state_dict()[f'transformer.h.{i}.ln_2.bias']

        # Remap the final layer norm values.
        our_model.final_layer_norm.weight.data = gpt_model.state_dict()['transformer.ln_f.weight']
        our_model.final_layer_norm.bias.data = gpt_model.state_dict()['transformer.ln_f.bias']

        return our_model