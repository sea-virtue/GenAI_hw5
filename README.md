# 清华大学校园问答助手微调项目

本项目用于“生成式人工智能第五次作业”：收集清华大学公开信息，构建指令微调数据集，并在 Qwen 小模型上进行 LoRA SFT，最后比较原始模型与微调模型的问答效果。

## 目录结构

```text
configs/high_quality_seed_urls.txt  # 推荐使用：服务、图书馆、校历、院系等高质量入口
configs/yxsz_seed_urls.txt          # 只围绕“院系设置”递归爬取
scripts/crawl_thu.py                # 爬虫，支持 HTML/PDF、编码修复、正文过滤
scripts/build_dataset.py            # 从网页文本生成 QA 数据并划分 train/eval
scripts/train_lora.py               # transformers + peft LoRA SFT
scripts/evaluate.py                 # 原始模型/微调模型评测
scripts/repair_mojibake.py          # 修复旧数据中的 UTF-8/GBK 乱码
src/thu_qa/                         # 通用工具
REPORT.md                           # 实验报告模板
requirements.txt                    # Python 依赖
```

## 1. 安装环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

如果服务器 CUDA 版本需要指定 PyTorch，请先按服务器环境安装官方 PyTorch，再安装其余依赖。

## 2. 爬取公开数据

推荐使用高质量种子，优先抓取图书馆、校园服务、校历、院系介绍、招生就业等公开页面：

```bash
python scripts/crawl_thu.py \
  --seeds configs/high_quality_seed_urls.txt \
  --output data/raw/thu_pages.jsonl \
  --max-pages 500 \
  --max-depth 3 \
  --delay 0.8
```

如果只想重点围绕“院系设置”递归抓院系官网：

```bash
python scripts/crawl_thu.py \
  --seeds configs/yxsz_seed_urls.txt \
  --output data/raw/yxsz_pages.jsonl \
  --max-pages 500 \
  --max-depth 3 \
  --delay 0.8
```

爬虫会把院系首页、图书馆首页等作为“链接发现页”。raw JSONL 中会保留一部分发现页和列表页，但会增加 `qa_candidate` 与 `page_role` 字段；构建训练集时默认只使用 `qa_candidate=true` 的页面。需要登录的内网页面、统一身份认证后的信息门户内容不会爬取。

## 3. 构建数据集

```bash
python scripts/build_dataset.py \
  --input data/raw/thu_pages.jsonl \
  --train-output data/processed/train.jsonl \
  --eval-output data/processed/eval.jsonl \
  --eval-size 50
```

如果想连 `qa_candidate=false` 的发现页也一起生成 QA，可加 `--include-low-quality`，但不建议用于最终训练。

作业要求训练集不少于 200 条，评测集不少于 50 条。建议总 QA 样本做到 1000-1500 条以上：

```bash
wc -l data/processed/train.jsonl
wc -l data/processed/eval.jsonl
```

如果数量不足，提高 `--max-pages`，或继续补充公开服务页、公开 PDF、校历、图书馆和院系介绍页。

## 4. 训练

```bash
python scripts/train_lora.py \
  --model-name Qwen/Qwen3-0.6B \
  --train-file data/processed/train.jsonl \
  --eval-file data/processed/eval.jsonl \
  --output-dir outputs/qwen3-0.6b-thuqa-lora \
  --load-in-4bit \
  --bf16 \
  --gradient-checkpointing \
  --num-train-epochs 3
```

如果 GPU 不支持 bf16，去掉 `--bf16`。

## 5. 评测

```bash
python scripts/evaluate.py \
  --model-name Qwen/Qwen3-0.6B \
  --adapter-dir outputs/qwen3-0.6b-thuqa-lora \
  --eval-file data/processed/eval.jsonl \
  --output-dir outputs/eval \
  --bf16
```

输出包括 `outputs/eval/metrics.json`、预测文件和失败案例文件。报告中需要比较原始模型与微调模型，并分析信息遗漏、幻觉、表达模糊、时间敏感信息处理不当等失败案例。

## 6. 一键流程

```bash
FORCE_CRAWL=1 bash run_pipeline.sh
```

默认种子文件是 `configs/high_quality_seed_urls.txt`。可以通过环境变量调整：

```bash
SEEDS=configs/yxsz_seed_urls.txt MAX_PAGES=800 MAX_DEPTH=3 FORCE_CRAWL=1 bash run_pipeline.sh
```

已有 raw 数据时，一键脚本默认复用，不会每次重新爬。若旧数据有乱码，可先尝试：

```bash
python scripts/repair_mojibake.py --input data/raw/thu_pages.jsonl --output data/raw/thu_pages_repaired.jsonl
```

但旧文件里的导航污染无法完美恢复，正式训练建议重新爬取。
