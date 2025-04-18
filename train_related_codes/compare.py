import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge import Rouge
import matplotlib.pyplot as plt

# ================= 配置部分 =================
MODEL_PATHS = {
    "base": "/home/lpy/Desktop/My_LLM/TinyLlama-1.1B-Chat-v0.6",   # 原始模型路径
    "tuned": "/home/lpy/Desktop/My_LLM/Cherry_Tinyllama"  # 微调后模型路径
}
TEST_DATA_PATH = "/home/lpy/Desktop/My_LLM/compare_alpaca.json"  # 测试数据集路径
NUM_EXAMPLES = 10                     # 评估样本数量
MAX_NEW_TOKENS = 1024                  # 生成最大长度
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============== 1. 加载模型和分词器 ==============
def load_models():
    models = {}
    tokenizers = {}
    
    for name, path in MODEL_PATHS.items():
        print(f"Loading {name} model...")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        models[name] = model
        tokenizers[name] = tokenizer
    
    return models, tokenizers

# ============== 2. 准备测试数据 ==============
def prepare_test_data():
    dataset = load_dataset("json", data_files=TEST_DATA_PATH, split="train")
    test_data = []
    
    for example in dataset.shuffle().select(range(NUM_EXAMPLES)):
        # 优化提示构建逻辑
        input_section = f"\n\n### Input:\n{example['input']}" if example['input'].strip() else ""
        prompt = f"""Below is an instruction describing a task. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}{input_section}

### Response:"""
        
        test_data.append({
            "prompt": prompt,
            "reference": example['output'],
            "instruction": example['instruction'],
            "input": example['input']
        })
    
    return test_data

# ============== 3. 生成文本函数 ==============
def generate_responses(models, tokenizers, test_data):
    results = []
    
    for example in tqdm(test_data, desc="Generating Responses"):
        record = example.copy()
        
        for model_name in MODEL_PATHS.keys():
            inputs = tokenizers[model_name](
                example["prompt"],
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(DEVICE)
            
            outputs = models[model_name].generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                do_sample=True
            )
            
            response = tokenizers[model_name].decode(
                outputs[0][len(inputs.input_ids[0]):], 
                skip_special_tokens=True
            )
            record[model_name] = response
        
        results.append(record)
    
    return results

# ============== 4. 自动评估指标 ==============
def calculate_metrics(results):
    # BLEU计算
    def calc_bleu(candidate, reference):
        return sentence_bleu(
            [reference.split()], 
            candidate.split(),
            weights=(0.5, 0.5, 0, 0)  # 使用BLEU-2
        )
    
    # ROUGE计算
    rouge = Rouge()
    
    # 初始化结果存储
    metrics = {
        name: {"bleu": [], "rouge-1": [], "rouge-2": [], "rouge-l": []}
        for name in MODEL_PATHS.keys()
    }
    
    # GPT-4评分（需要API密钥）
    gpt_scores = None  # 需要自行实现API调用
    
    # 计算指标
    for result in results:
        for model_name in MODEL_PATHS.keys():
            candidate = result[model_name]
            reference = result["reference"]
            
            # BLEU
            metrics[model_name]["bleu"].append(calc_bleu(candidate, reference))
            
            # ROUGE
            if candidate.strip() and reference.strip():
                scores = rouge.get_scores(candidate, reference)[0]
                metrics[model_name]["rouge-1"].append(scores['rouge-1']['f'])
                metrics[model_name]["rouge-2"].append(scores['rouge-2']['f'])
                metrics[model_name]["rouge-l"].append(scores['rouge-l']['f'])
    
    # 聚合结果
    final_metrics = {}
    for name in MODEL_PATHS.keys():
        final_metrics[name] = {
            "BLEU-2": np.mean(metrics[name]["bleu"]),
            "ROUGE-1": np.mean(metrics[name]["rouge-1"]),
            "ROUGE-2": np.mean(metrics[name]["rouge-2"]),
            "ROUGE-L": np.mean(metrics[name]["rouge-l"])
        }
    
    return final_metrics

# ============== 5. 可视化对比 ==============
def visualize_comparison(metrics):
    df = pd.DataFrame(metrics).T
    ax = df.plot(kind='bar', figsize=(10,6), 
                title="Model Performance Comparison")
    plt.xticks(rotation=0)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    # plt.show()

# ============== 6. 示例对比展示 ==============
def show_examples(results, num=10):
    print("\n=== 示例输出对比 ===")
    for i, example in enumerate(results[:num]):
        print(f"\n示例 {i+1}")
        print(f"指令：{example['instruction']}")
        print(f"输入：{example['input']}")

        for model in MODEL_PATHS.keys():
            print(f"{model} 响应：{example[model]}")

# ============== 主执行流程 ==============
if __name__ == "__main__":
    # 1. 加载资源
    models, tokenizers = load_models()
    test_data = prepare_test_data()
    
    # 2. 生成响应
    results = generate_responses(models, tokenizers, test_data)
    
    # 3. 计算指标
    metrics = calculate_metrics(results)
    print("\n=== 自动评估指标 ===")
    print(pd.DataFrame(metrics).T.round(3))
    
    # 4. 可视化
    visualize_comparison(metrics)
    
    # 5. 显示示例
    show_examples(results)
    
    # 6. 保存结果
    with open("eval_results.json", "w") as f:
        json.dump({"metrics": metrics, "examples": results[:5]}, f, indent=2)