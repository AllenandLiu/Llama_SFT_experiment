# Llama_SFT_experiment
“大模型：算法与实践”课程实验项目代码

本次实验主要内容是进行一次SFT数据集清洗和筛选，并用选择的数据来对大模型进行一次训练。

## 项目结构
文件目录如下：

```text
Llama_SFT_experiment
├── Cherry_Tinyllama/
│   ├── config.json
|   ├── generation_config.json
|   ├── model.safetensors_part_*
|   ├── special_tokens_map.json
|   ├── tokenizer.json
|   ├── tokenizer.model
|   ├── tokenizer_config.json
|   └── training_args.bin
├── TinyLlama-1.1B-Chat-v0.6/
│   ├── config.json
│   ├── generation_config.json
│   ├── ggml-model-q4_0.gguf_part_*
|   ├── model.safetensors_part_*
|   ├── special_tokens_map.json
|   ├── tokenizer.json
|   ├── tokenizer.model
|   └── tokenizer_config.json
├── cherry_selection/
│   ├── data_analysis.py
|   ├── data_by_IFD.py
|   └── data_by_cluster.py
├── data/
|   ├── alpaca.pt_part_*
|   ├── alpaca_data.json
|   ├── cherry_alpaca.json
|   └── compare_alpaca.json
├── train_related_codes/
│   ├── compare.py
|   ├── compare_log.txt
|   ├── eval_results.json
|   ├── model_comparison.png
|   ├── train.py
|   └── train_log.txt
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

Cherry_Tinyllama：该目录下是作者通过微调得到的模型，由于github上传限制的原因，所以部分大文件通过Git BASH的split命令进行了分割，可以通过cat指令来将文件合并，示例如下：

```bash
cat  model.safetensors_part_* > model.safetensors
```

TinyLlama-1.1B-Chat-v0.6：是本次实验所采用的Base模型。

cherry_selection：是SFT数据清洗筛选的方法，其中包含data_analysis，data_by_IFD，data_by_cluster三个代码，在使用的时候需要修改相应的路径或根据需要修改部分参数，示例如下：

```bash
python ./data_analysis.py --_data_path "your_jason_data" --save_path "touy_jason-save_path" --model_name_or_path "your_model" --prompt alpaca --mod cherry
```

具体的参数含义可以自行参考代码内容。

数据清洗和筛选的方法出自论文“[From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning](https://arxiv.org/abs/2308.12032)”，这篇论文提出了提出一个指令跟随难度（Instruction-Following Difficulty，IFD）指标，通过该指标来筛选具有增强LLM指令调优潜力的数据样例（樱桃数据，cherry data），而模型仅使用原始数据5%-10%的cherry数据就可以达到全量数据微调的效果。论文相关的内容可自行阅读论文或访问其[代码仓库](https://github.com/MingLiiii/Cherry_LLM)进行了解。

data：该目录下是本实验中用到的相关数据，包括原始数据alpaca_data.json，经过分析计算得到的包含IFD评分等内容的alpaca.pt，清洗后的数据cherry_alpaca.json以及通过Kimi生成的模型效果对比数据compare_alpaca.json。其中原始数据alpaca_data.json来源是斯坦福的[Alpaca项目](https://github.com/tatsu-lab/stanford_alpaca)。

train_related_codes：该目录下是一些作者自行编写的代码，包括模型训练代码train.py，模型效果对比代码compare.py，以及相应的日志文件和结果。

## 快速开始
本次实验作者采用的是python=3.8，建议通过conda创建虚拟环境来管理环境。

requirements.txt中包含了本次实验所需要的依赖库，通过pip安装即可。
```python
pip install -r requirements.txt
```
