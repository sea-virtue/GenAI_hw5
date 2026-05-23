# 清华大学校园问答助手微调项目

本项目用于“生成式人工智能第五次作业”：收集清华大学校园公开信息，构建指令微调数据集，并在 Qwen 小模型上做 LoRA SFT，最后比较原始模型与微调模型的问答效果。

## 目录结构

```text
configs/crawl_seed_urls.txt   # 爬虫种子 URL，可继续添加院系/部门/图书馆/PDF 页面
scripts/crawl_thu.py          # 爬取清华公开网页，支持 HTML 和 PDF
scripts/build_dataset.py      # 从网页文本生成指令问答数据，划分 train/eval
scripts/train_lora.py         # transformers + peft LoRA SFT 训练
scripts/evaluate.py           # 原始模型/微调模型自动评测与失败案例导出
src/thu_qa/                   # 通用读写、清洗、指标函数
REPORT.md                     # 实验报告模板
requirements.txt              # Python 依赖
```

## 1. 安装环境

建议在 Linux GPU 服务器上运行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

如果服务器 CUDA 版本需要指定 PyTorch，请先按服务器环境安装官方 PyTorch，再安装 `requirements.txt` 中其余依赖。

## 2. 爬取公开网页

先检查 `configs/crawl_seed_urls.txt`，可以继续加入课程、图书馆、后勤、院系、通知、公开 PDF 等页面。

```bash
python scripts/crawl_thu.py \
  --seeds configs/crawl_seed_urls.txt \
  --output data/raw/thu_pages.jsonl \
  --max-pages 120 \
  --max-depth 2 \
  --delay 0.8
```

输出文件每行是一条网页记录，字段包括 `url`、`title`、`source_type`、`text`、`crawled_at`。爬虫会限制在清华相关域名内，并读取 `robots.txt`。

## 3. 构建训练集和评测集

```bash
python scripts/build_dataset.py \
  --input data/raw/thu_pages.jsonl \
  --train-output data/processed/train.jsonl \
  --eval-output data/processed/eval.jsonl \
  --eval-size 50
```

建议最终数据量：

```bash
wc -l data/processed/train.jsonl
wc -l data/processed/eval.jsonl
```

训练集不少于 200 条，评测集不少于 50 条。如果数量不足，增加 `configs/crawl_seed_urls.txt` 中的种子页面，或提高 `--max-pages`。

数据格式示例：

```json
{"id":"thuqa-00001","instruction":"请概括介绍清华大学图书馆。","input":"","output":"...","source_title":"...","source_url":"..."}
```

## 4. LoRA SFT 训练

显存较紧时建议启用 4bit：

```bash
python scripts/train_lora.py \
  --model-name Qwen/Qwen3-0.6B \
  --train-file data/processed/train.jsonl \
  --eval-file data/processed/eval.jsonl \
  --output-dir outputs/qwen3-0.6b-thuqa-lora \
  --load-in-4bit \
  --bf16 \
  --gradient-checkpointing \
  --num-train-epochs 3 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8
```

如果 GPU 不支持 bf16，去掉 `--bf16`。训练产物会保存在 `outputs/qwen3-0.6b-thuqa-lora`，其中是 LoRA adapter 和 tokenizer。

## 5. 性能评测

脚本会依次评测原始模型和微调模型，输出自动指标与失败案例：

```bash
python scripts/evaluate.py \
  --model-name Qwen/Qwen3-0.6B \
  --adapter-dir outputs/qwen3-0.6b-thuqa-lora \
  --eval-file data/processed/eval.jsonl \
  --output-dir outputs/eval \
  --bf16
```

主要输出：

```text
outputs/eval/metrics.json
outputs/eval/base_predictions.jsonl
outputs/eval/finetuned_predictions.jsonl
outputs/eval/base_failure_cases.jsonl
outputs/eval/finetuned_failure_cases.jsonl
```

自动指标包括：

- `exact_match`：预测与参考答案完全一致的比例。
- `contains_reference`：预测是否包含参考答案。
- `char_f1`：中文字符级 F1，更适合开放式短问答。

报告里还需要人工分析若干失败案例，重点看信息遗漏、幻觉、表达模糊、时间敏感信息处理不当等问题。

## 6. 一键流程

也可以直接运行：

```bash
bash run_pipeline.sh
```

默认会复用已有的 `data/raw/thu_pages.jsonl`，只有该文件不存在或为空时才重新爬取。若想强制重新爬取：

```bash
FORCE_CRAWL=1 bash run_pipeline.sh
```

如果只想重新训练，不想重新构建数据，可以直接跳过一键脚本，运行第 4、5 步的训练和评测命令。训练前请确认服务器能访问 Hugging Face 或已经缓存好 `Qwen/Qwen3-0.6B`。
