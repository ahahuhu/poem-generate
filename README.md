# 中文诗歌生成实验（基于GPT2）

本项目为重庆大学自然语言处理课程实验，旨在基于 GPT2 预训练模型实现中文诗歌的生成。项目支持加载 HuggingFace 预训练权重，进行多种微调训练，并提供多种生成策略（贪心、top-k、top-p、beam search、藏头诗等），并将训练好的模型在ROUGE1和GOUGE2上面进行评估。

## 目录结构

```bash
poem-generate/
├── config.py                # 配置文件，定义模型结构参数
├── datasets.py              # 数据集处理与加载
├── model/
│   ├── gpt2.py              # GPT2模型结构与权重加载
│   ├── poem_gpt.py          # 封装诗歌生成模型与生成方法
│   └── base_gpt             # 存储GPTPreTrainedModel的配置文件
├── scripts/
│   ├── run_experiments.py   # 实验批量运行脚本
│   └── generate_res.py      # 生成诗歌结果脚本
├── train.py                 # 训练主程序
├── generate.py              # 交互式生成诗歌脚本
├── test.py                  # 评测与ROUGE分数计算
├── data/                    # 数据集与中间数据
├── result/                  # 训练日志与生成结果
├── main.ipynb               # 数据分析与预处理notebook
├── exp4.ipynb               # 生成与评测实验notebook
├── ori.ipynb                # 原始的基于LSTM生成模型notebook
└── test.ipynb               # 生成与评测notebook
```

## 环境依赖

- Python 3.8+
- torch
- transformers
- einops
- tqdm
- numpy
- jieba
- evaluate
- 其它见 requirements.txt（如有）

## 数据准备

1. **原始数据**：请将清洗后的诗歌数据保存为 `data/clear_data.npy`。
2. **分词器**：首次运行会自动下载并缓存 BERT 分词器到 `cache/bert-tokenizer`。
3. **预训练模型**：首次运行会自动下载 GPT2 中文预训练权重到 `cache/pretrained_model`。

## 训练模型

```bash
python train.py --epochs 20 --batch_size 16 --lr 1e-4 --use_gpu
```

- 支持参数：`--poem_path`、`--epochs`、`--batch_size`、`--lr`、`--use_gpu`、`--use_lora` 等。
- 训练日志与loss会保存在 `result/` 目录下。

## 生成诗歌

### 交互式生成

```bash
python generate.py --use_gpu
```

输入诗句前缀，模型将自动生成后续内容。导入不同的模型可以生成不同类型的诗歌。也支持不同的生成策略的指定。

### 批量生成

可参考 `scripts/generate_res.py`，批量生成多首诗歌并保存。

## 生成方法

- **贪心搜索**（greedy_search）
- **Top-k采样**（generate_top_k）
- **Top-p采样**（generate_top_q）
- **Beam Search**（generate_beam_search）
- **藏头诗生成**（generate_hidden_poem）

详见 [`model/poem_gpt.py`](model/poem_gpt.py) 中各生成方法实现。

## 评测方法

- 支持字级别、词级别（jieba分词）ROUGE-1/2 分数评测。
- 运行 `test.py` 自动评测不同生成方法的结果。

## 参考/致谢

- 重庆大学自然语言处理课程
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [UER GPT2-Chinese](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall)
- 华为云 ModelArt 平台

## 联系方式

如有问题请联系课程助教或在 Issues 区留言。

---

**实验目的**：掌握预训练语言模型的微调与文本生成方法，提升中文自然语言生成的工程实践能力。
