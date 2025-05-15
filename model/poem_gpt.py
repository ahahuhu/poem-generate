import heapq
import torch
from torch import nn
from model.gpt2 import GPT2Model
from transformers import BertTokenizer

class PoemGPT(nn.Module):

  def __init__(self, args, tokenizer):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model_name=args.model_name, model_dir=args.model_dir, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = tokenizer

    # 将最终的输出last_hidden_state转化为词汇表的概率分布
    self.vob_proj = nn.Linear(args.d, self.tokenizer.vocab_size)
    self.config = args
    self.device = "cuda" if args.use_gpu else "cpu"

    for name, param in self.gpt.named_parameters():
        param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    sequence_output, last_token = self.gpt(input_ids, attention_mask).values()
    return self.vob_proj(sequence_output)


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate_top_q(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    """top-p(nucleus)"""
    token_ids = encoding.to(self.get_device())
    # 1表示正常的输入 0表示pad的值
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())
    for _ in range(max_length):
      # Forward pass to get logits
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )
      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.sep_token_id:
        break

    # 去除前3个token，一般前3个token为特殊的token，比如说[CLS]、[BOS]
    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())
    return token_ids, generated_output
  
  @torch.no_grad()
  def generate_top_k(self, encoding, temperature=0.7, k_size=3, max_length=128):
    """top-k"""
    token_ids = encoding.to(self.get_device()) #(bs, sl)
    # 1表示正常的输入 0表示pad的值
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())
    for _ in range(max_length):
      # Forward pass to get logits
      logits_sequence = self.forward(token_ids, attention_mask) #(bs,sl,vocab_size)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling (bs, vocab_size)

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Top-k sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      filtered_probs = sorted_probs[:k_size] # 取出前k个最大的Token
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )
      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.sep_token_id:
        break

    # 去除前3个token，一般前3个token为特殊的token，比如说[CLS]、[BOS]
    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())
    return token_ids, generated_output

  @torch.no_grad()
  def generate_greedy_search(self, encoding, max_length=128):
    """greedy_search"""
    token_ids = encoding.to(self.get_device()) #(bs, sl)
    # 1表示正常的输入 0表示pad的值
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())
    for _ in range(max_length):
      # Forward pass to get logits
      logits_sequence = self.forward(token_ids, attention_mask) #(bs,sl,vocab_size)
      last_logits = logits_sequence[:, -1, :]
      # greedy search 直接取出最大的一个值
      sampled_token_id = torch.argmax(last_logits, dim=1).unsqueeze(dim=0)

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token_id], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )
      # Stop if end-of-sequence token is reached
      if sampled_token_id.item() == self.tokenizer.sep_token_id:
        break

    # 去除前3个token，一般前3个token为特殊的token，比如说[CLS]、[BOS]
    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())
    return token_ids, generated_output

  @torch.no_grad()
  def generate_beam_search(self, encoding, beam_size:int = 3, max_length=128):
    """beam_search"""
    token_ids = encoding.to(self.get_device()) #(bs, sl)
    # 1表示正常的输入 0表示pad的值
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())
    # beam search需要一个beams来保存最佳的路线
    beams = [(token_ids, 1)]
    for _ in range(max_length):
      t_res = [] # 存放备份的token序列
      for token_id, score in beams:
        if token_id[0, -1].item() == self.tokenizer.sep_token_id:
          t_res.append((token_id, score))
          continue
        logits_sequence = self.forward(token_id, attention_mask) #(bs,sl,vocab_size)
        last_logits = logits_sequence[:, -1, :]
        # 根据last_prob进行排序，取出前beam size大小的数量
        last_logits, indices  = torch.topk(last_logits, beam_size, -1)  # (bs, beam_size)
        last_scores = torch.nn.functional.softmax(last_logits, -1)
        for j in range(beam_size):
          # 更新每条路径的分数和token
          t_res.append((torch.cat([token_id, indices[0, j].unsqueeze(0).unsqueeze(0)], dim=1),
                         torch.log(last_scores[0][j]).item() +score))
      # Append sampled token
      beams = sorted(t_res, key=lambda x: -x[1])[:beam_size]
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    # 去除前3个token，一般前3个token为特殊的token，比如说[CLS]、[BOS]
    generated_output = self.tokenizer.decode(beams[0][0][0].cpu().numpy().tolist())
    return token_ids, generated_output

  @torch.no_grad()
  def generate_hidden_poem(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    """首字引导式GPT-2藏头生成法，每次生成时输入为之前所有已生成诗句+当前首字"""
    device = self.get_device()
    token_ids = encoding.to(device)  # shape: (1, N)
    generated_token_ids = [encoding[0][0].unsqueeze(0)]

    for idx in range(1, token_ids.shape[1]):
      # 拼接之前所有已生成的token和当前首字
      if generated_token_ids:
        prev_tokens = [tid for sent in generated_token_ids for tid in sent]
        prev_tokens_tensor = torch.tensor(prev_tokens, dtype=torch.long, device=device).unsqueeze(0)
        cur_ids = torch.cat([prev_tokens_tensor, token_ids[:, idx:idx+1]], dim=1)
      else:
        cur_ids = token_ids[:, idx:idx+1]  # 第一句只用首字

      cur_sentence = cur_ids.clone()
      for _ in range(max_length):
        token_ids_out, _ = self.generate_top_q(cur_sentence, temperature, top_p, max_length=1)
        next_token = token_ids_out[0, -1].unsqueeze(0).unsqueeze(0)
        cur_sentence = torch.cat([cur_sentence, next_token], dim=1)
        # 判断是否为句末
        if next_token.item() == self.tokenizer.sep_token_id or self.tokenizer.decode([next_token.item()]) == '。':
          break
      # 只保留本次新生成的内容（去掉输入部分，只保留首字及后续生成）
      if generated_token_ids:
        # 新生成部分为 cur_sentence 去掉前面 prev_tokens 的部分
        new_tokens = cur_sentence[0, len(prev_tokens):].tolist()
      else:
        new_tokens = cur_sentence[0].tolist()
      generated_token_ids.append(new_tokens)

    # 拼接所有句子的token
    all_token_ids = []
    for ids in generated_token_ids:
      all_token_ids.extend(ids)
    all_token_ids.append(self.tokenizer.sep_token_id)
    all_token_ids = torch.tensor(all_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated_output = self.tokenizer.decode(all_token_ids[0].cpu().numpy().tolist())
    return all_token_ids, generated_output