from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer  # 移除了SFTConfig
import torch

# 参数配置（修复Windows路径反斜杠问题）
model_path = "/home/lpy/Desktop/My_LLM/TinyLlama-1.1B-Chat-v0.6"
dataset_path = "/home/lpy/Desktop/My_LLM/cherry_alpaca.json"
output_dir = "/home/lpy/Desktop/My_LLM/Cherry_Tinyllama"
max_seq_length = 512

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# 数据处理函数（保持不变）
def format_alpaca(example):
    instruction = example['instruction']
    input_text = example['input']
    output = example['output']
    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}" if input_text else f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    return {"text": text}

# 加载数据集
dataset = load_dataset('json', data_files=dataset_path, split='train').map(format_alpaca)

# 训练参数配置（旧版参数传递方式）
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    optim="adamw_torch",
    report_to="none"
)

# 初始化训练器（直接传递参数）
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",      # 直接传递参数
    max_seq_length=max_seq_length,  # 直接传递参数
    packing=True,                  # 直接传递参数
    tokenizer=tokenizer
)

# 开始训练
trainer.train()
trainer.save_model(output_dir)