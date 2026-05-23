# 实验报告模板

## 1. 数据集构建

- 数据来源：清华大学官网、图书馆、信息门户、新闻网、就业等公开页面。
- 爬取方式：使用 `scripts/crawl_thu.py` 从 `configs/crawl_seed_urls.txt` 出发，限制在清华相关域名内，支持 HTML/PDF 文本抽取，并进行 URL 去重、文本去重和最小长度过滤。
- 数据清洗：去除导航、页脚、脚本、重复行和过短文本；将长文本按句号附近切分成适合问答的片段。
- 问答生成：使用标题概括类问题、地点/时间/联系方式/办理流程等关键词模板构造指令问答。
- 数据规模：
  - 训练集：填写 `wc -l data/processed/train.jsonl` 的结果。
  - 评测集：填写 `wc -l data/processed/eval.jsonl` 的结果。

## 2. 模型与训练设置

- 基座模型：`Qwen/Qwen3-0.6B`
- 微调方法：监督微调 SFT + LoRA
- 训练脚本：`scripts/train_lora.py`
- 主要超参数：
  - epoch：3
  - LoRA rank：16
  - LoRA alpha：32
  - 学习率：2e-4
  - max length：1024
  - batch size：按实际填写
  - gradient accumulation：按实际填写
  - 量化：是否使用 4bit，按实际填写

## 3. 评测方法

- 评测脚本：`scripts/evaluate.py`
- 自动指标：
  - exact match
  - contains reference
  - character-level F1
- 对比对象：
  - 原始 `Qwen/Qwen3-0.6B`
  - LoRA 微调后的校园问答模型

## 4. 实验结果

将 `outputs/eval/metrics.json` 中的结果整理成表格：

| 模型 | Exact Match | Contains Reference | Char F1 |
| --- | ---: | ---: | ---: |
| 原始模型 | 填写 | 填写 | 填写 |
| 微调模型 | 填写 | 填写 | 填写 |

## 5. 案例分析

### 成功案例

从 `outputs/eval/finetuned_predictions.jsonl` 中选择 2-3 条，说明微调模型相比原始模型在哪些方面更准确，例如能引用校园服务细节、回答更聚焦、减少泛化表达。

### 失败案例

从 `outputs/eval/finetuned_failure_cases.jsonl` 中选择 3-5 条，分析失败原因：

- 信息遗漏：答案只覆盖了参考答案的一部分。
- 幻觉：模型补充了数据集中没有依据的信息。
- 表达模糊：回答过于笼统，没有给出具体地点、时间、流程或联系方式。
- 时间敏感信息：通知、开放时间等内容可能更新，需要重新爬取最新页面。

## 6. 总结

总结数据规模、训练效果、微调收益和不足。可以说明后续改进方向，例如加入人工审核问答、增加院系和职能部门数据、引入 RAG 检索增强、对时间敏感信息定期更新。
