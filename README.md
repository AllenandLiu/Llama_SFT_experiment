# Llama_SFT_experiment
“大模型：算法与实践”课程实验项目代码

本次实验主要内容是进行一次SFT数据集清洗和筛选，并用选择的数据来对大模型进行一次训练。

##项目结构
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

