# 清华大学校园问答助手微调项目

本项目用于“生成式人工智能第五次作业”：爬取清华大学公开网页，构造校园问答数据集，在 Qwen 小模型上进行 LoRA SFT，并比较原始模型与微调模型的问答效果。

## 目录结构

```text
configs/high_quality_seed_urls.txt  # 推荐入口：服务、图书馆、校历、院系等高质量页面
configs/yxsz_seed_urls.txt          # 围绕“院系设置”递归爬取
configs/crawl_seed_urls.txt         # 早期通用入口
scripts/crawl_thu.py                # 爬虫：HTML/PDF、编码修复、正文抽取、链接发现
scripts/build_dataset.py            # 从 raw 网页生成 QA 数据，并划分 train/eval
scripts/train_lora.py               # transformers + peft LoRA SFT
scripts/evaluate.py                 # 原始模型/微调模型自动评测
scripts/chat_lora.py                # 交互式查看 base 或 LoRA 模型效果
scripts/repair_mojibake.py          # 修复旧数据中的乱码
src/thu_qa/                         # 通用工具
REPORT.md                           # 实验报告模板
requirements.txt                    # Python 依赖
```

## 1. 安装环境

服务器上建议使用 conda 环境，并始终用 `python -m pip` 安装到当前环境：

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

如果服务器无法直连 Hugging Face，训练和评测脚本默认使用 `https://hf-mirror.com`。

## 2. 爬取公开数据

推荐使用高质量种子，优先爬取图书馆、校园服务、校历、院系介绍、招生就业等公开页面：

```bash
python scripts/crawl_thu.py \
  --seeds configs/high_quality_seed_urls.txt \
  --output data/raw/thu_pages.jsonl \
  --max-pages 800 \
  --max-depth 3 \
  --delay 0.8
```

如果只想围绕“院系设置”递归爬取：

```bash
python scripts/crawl_thu.py \
  --seeds configs/yxsz_seed_urls.txt \
  --output data/raw/yxsz_pages.jsonl \
  --max-pages 800 \
  --max-depth 3 \
  --delay 0.8
```

爬虫会保留部分首页、列表页作为链接发现页，但会写入 `qa_candidate` 和 `page_role` 字段。构建数据集时默认只使用 `qa_candidate=true` 的页面。需要登录、内网或统一身份认证后的内容不会爬取。

## 3. 构建数据集

```bash
python scripts/build_dataset.py \
  --input data/raw/thu_pages.jsonl \
  --train-output data/processed/train.jsonl \
  --eval-output data/processed/eval.jsonl \
  --eval-size 50
```

作业要求训练集不少于 200 条、评测集不少于 50 条。检查数量：

```bash
wc -l data/processed/train.jsonl
wc -l data/processed/eval.jsonl
```

如果数量不足，提高 `--max-pages`，或补充更多公开服务页面、PDF、校历、图书馆和院系介绍页。

## 4. 训练 LoRA

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/train_lora.py \
  --model-name Qwen/Qwen3-0.6B \
  --train-file data/processed/train.jsonl \
  --eval-file data/processed/eval.jsonl \
  --output-dir outputs/qwen3-0.6b-thuqa-lora \
  --load-in-4bit \
  --bf16 \
  --gradient-checkpointing \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --max-length 1024 \
  --num-train-epochs 3
```

如果 GPU 不支持 bf16，去掉 `--bf16`。

## 5. 自动评测

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/evaluate.py \
  --model-name Qwen/Qwen3-0.6B \
  --adapter-dir outputs/qwen3-0.6b-thuqa-lora \
  --eval-file data/processed/eval.jsonl \
  --output-dir outputs/eval \
  --bf16
```

输出包括：

```text
outputs/eval/metrics.json
outputs/eval/base_predictions.jsonl
outputs/eval/finetuned_predictions.jsonl
outputs/eval/base_failure_cases.jsonl
outputs/eval/finetuned_failure_cases.jsonl
```

注意：网页摘要型答案很难逐字匹配，`exact_match` 和 `contains_reference` 往往偏低。报告中应结合 `char_f1` 和人工样例分析。

## 6. 交互式聊天

用 HF/PEFT 直接加载 base + LoRA：

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/chat_lora.py \
  --model-name Qwen/Qwen3-0.6B \
  --adapter-dir outputs/qwen3-0.6b-thuqa-lora \
  --load-in-4bit \
  --bf16 \
  --no-thinking
```

如果复制多行命令时终端一直换行、没有开始加载模型，说明 shell 还在等待续行。可以改用单行命令：

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/chat_lora.py --model-name Qwen/Qwen3-0.6B --adapter-dir outputs/qwen3-0.6b-thuqa-lora --load-in-4bit --bf16 --no-thinking
```

如果 Qwen3 输出较长思考内容，可以关闭 thinking 模式：

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/chat_lora.py \
  --model-name Qwen/Qwen3-0.6B \
  --adapter-dir outputs/qwen3-0.6b-thuqa-lora \
  --load-in-4bit \
  --bf16 \
  --no-thinking
```

对比原始 base 模型：

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/chat_lora.py \
  --model-name Qwen/Qwen3-0.6B \
  --adapter-dir "" \
  --load-in-4bit \
  --bf16
```

## 7. vLLM 服务方式

如果服务器安装了 vLLM，可以用 OpenAI 兼容接口启动：

```bash
CUDA_VISIBLE_DEVICES=4 vllm serve Qwen/Qwen3-0.6B \
  --enable-lora \
  --lora-modules thuqa=outputs/qwen3-0.6b-thuqa-lora \
  --served-model-name thuqa \
  --dtype bfloat16 \
  --port 8000
```

另开终端请求：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"thuqa","messages":[{"role":"user","content":"清华大学校园参观如何预约？"}],"temperature":0,"max_tokens":256}'
```

vLLM 对 LoRA、量化和版本兼容要求更高；如果只是查看效果，优先使用 `scripts/chat_lora.py`。

## 8. 一键流程

```bash
FORCE_CRAWL=1 bash run_pipeline.sh
```

默认种子文件是 `configs/high_quality_seed_urls.txt`。可以通过环境变量调整：

```bash
SEEDS=configs/yxsz_seed_urls.txt MAX_PAGES=800 MAX_DEPTH=3 FORCE_CRAWL=1 bash run_pipeline.sh
```

已有 raw 数据时，一键脚本默认复用，不会每次重新爬。若旧数据有乱码，可先尝试：

```bash
python scripts/repair_mojibake.py \
  --input data/raw/thu_pages.jsonl \
  --output data/raw/thu_pages_repaired.jsonl
```

如果旧文件的正文已经严重污染，正式训练建议重新爬取。
