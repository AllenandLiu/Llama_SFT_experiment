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
TinyLlama-1.1B-Chat-v0.6：是本次实验所采用的Base模型
cherry——selection：是SFT数据清洗筛选的方法，其中包含data_analysis，data_by_IFD，data_by_cluster三个代码，
