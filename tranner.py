# train_adapter.py
import os
import ast
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from peft import get_peft_model, LoraConfig

# 1. 模型仓库 ID（远程加载，无需本地 clone）
model_id = "unsloth/DeepSeek-R1-0528-Qwen3-8B"

# 2. 加载 tokenizer 和 模型（自动下载 sharded safetensors）
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)
# 使用默认 dtype（FP32）以兼容 Mac M-series 无 GPU 支持情况
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    trust_remote_code=True
)

# 3. 注入 LoRA Adapter
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none"
)
model = get_peft_model(model, peft_config)

# 强制将模型置于 CPU（macOS 无 GPU 支持）
model.to("cpu")

# 4. 自定义 ABSA 数据集
class ABSADataset(torch.utils.data.Dataset):
    def __init__(self, filepath, tokenizer, max_length=512):
        self.examples = []
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '####' not in line:
                    continue
                text, label_str = line.split('####', 1)
                labels = ast.literal_eval(label_str)
                prompt = f"评论: {text}####"
                response = str(labels)
                full = prompt + response
                tokenized = tokenizer(
                    full,
                    truncation=True,
                    max_length=max_length,
                    padding='max_length'
                )
                tokenized['labels'] = tokenized['input_ids'].copy()
                self.examples.append(tokenized)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.examples[idx].items()}

# 5. 加载数据集文件
train_dataset = ABSADataset('train.txt', tokenizer)
dev_dataset   = ABSADataset('dev.txt', tokenizer)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# 6. 训练参数配置（移除 bf16 以兼容无 GPU 环境）
training_args = TrainingArguments(
    output_dir='./lora_adapter_output',
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    eval_strategy='steps',
    eval_steps=300,
    logging_steps=50,
    num_train_epochs=3,
    learning_rate=2e-4,
    save_total_limit=2,
    save_steps=2,
    push_to_hub=False
)

# 7. 检查是否恢复训练
last_checkpoint = None
if os.path.isdir('./lora_adapter_output/checkpoint-936'):
    last_checkpoint = './lora_adapter_output/checkpoint-936'
    print(f"检测到检查点，准备从 {last_checkpoint} 恢复训练...")

# 8. 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=data_collator
)

# 9. 开始训练（支持断点恢复）
trainer.train(resume_from_checkpoint=last_checkpoint)

# 10. 保存最终模型和tokenizer
os.makedirs('./lora_adapter_output', exist_ok=True)
model.save_pretrained('./lora_adapter_output')
tokenizer.save_pretrained('./lora_adapter_output')
print("✅ LoRA adapter 已保存到 ./lora_adapter_output")

# # 7. 初始化 Trainer 并开始训练
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=dev_dataset,
#     data_collator=data_collator
# )

# trainer.train()

# # 8. 保存 LoRA Adapter 和 tokenizer
# os.makedirs('./lora_adapter_output', exist_ok=True)
# model.save_pretrained('./lora_adapter_output')
# tokenizer.save_pretrained('./lora_adapter_output')
# print("LoRA adapter 已保存到 ./lora_adapter_output")

# 9. 推理示例
if __name__ == '__main__':
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        trust_remote_code=True
    )
    adapter = PeftModel.from_pretrained(base, './lora_adapter_output')
    adapter.eval()
    text = "这家酒店的房间很舒服,但是服务人员态度很差."
    prompt = f"评论: {text}####"
    inputs = tokenizer(prompt, return_tensors="pt").to(adapter.device)
    out = adapter.generate(**inputs, max_new_tokens=128)
    result = tokenizer.decode(out[0], skip_special_tokens=True)
    print("推理结果:", result)


# # train_adapter.py
# import os
# import ast
# import torch
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     DataCollatorForLanguageModeling,
#     Trainer,
#     TrainingArguments
# )
# from peft import get_peft_model, LoraConfig

# # 1. 模型仓库 ID
# model_id = "unsloth/DeepSeek-R1-0528-Qwen3-8B"

# # 2. 加载 tokenizer 和模型
# tokenizer = AutoTokenizer.from_pretrained(
#     model_id,
#     trust_remote_code=True
# )
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     trust_remote_code=True,
#     device_map="cpu"  # Mac 无 GPU 使用 CPU
# )

# # 3. 注入 LoRA Adapter
# peft_config = LoraConfig(
#     task_type="CAUSAL_LM",
#     inference_mode=False,
#     r=8,
#     lora_alpha=16,
#     lora_dropout=0.05,
#     target_modules=["q_proj", "v_proj"],
#     bias="none"
# )
# model = get_peft_model(model, peft_config)
# model.to("cpu")  # 明确置于 CPU

# # 4. 自定义 ABSA 数据集
# class ABSADataset(torch.utils.data.Dataset):
#     def __init__(self, filepath, tokenizer, max_length=512):
#         self.examples = []
#         with open(filepath, encoding='utf-8') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line or '####' not in line:
#                     continue
#                 text, label_str = line.split('####', 1)
#                 labels = ast.literal_eval(label_str)
#                 prompt = f"评论: {text}####"
#                 response = str(labels)
#                 full = prompt + response
#                 tokenized = tokenizer(
#                     full,
#                     truncation=True,
#                     max_length=max_length,
#                     padding='max_length'
#                 )
#                 tokenized['labels'] = tokenized['input_ids'].copy()
#                 self.examples.append(tokenized)

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, idx):
#         return {k: torch.tensor(v) for k, v in self.examples[idx].items()}

# # 5. 加载数据
# train_dataset = ABSADataset('train.txt', tokenizer)
# dev_dataset   = ABSADataset('dev.txt', tokenizer)
# data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# # 6. 训练参数
# training_args = TrainingArguments(
#     output_dir='./lora_adapter_output',
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     gradient_accumulation_steps=4,
#     evaluation_strategy='steps',
#     eval_steps=100,
#     logging_steps=50,
#     num_train_epochs=3,
#     learning_rate=2e-4,
#     save_total_limit=2,
#     save_steps=8,  # 每8步保存
#     push_to_hub=False
# )

# # 7. 检查是否恢复训练
# last_checkpoint = None
# if os.path.isdir('./lora_adapter_output/checkpoint-last'):
#     last_checkpoint = './lora_adapter_output/checkpoint-last'
#     print(f"检测到检查点，准备从 {last_checkpoint} 恢复训练...")

# # 8. 初始化 Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=dev_dataset,
#     data_collator=data_collator
# )

# # 9. 开始训练（支持断点恢复）
# trainer.train(resume_from_checkpoint=last_checkpoint)

# # 10. 保存最终模型和tokenizer
# os.makedirs('./lora_adapter_output', exist_ok=True)
# model.save_pretrained('./lora_adapter_output')
# tokenizer.save_pretrained('./lora_adapter_output')
# print("✅ LoRA adapter 已保存到 ./lora_adapter_output")

# # 11. 推理测试（可选）
# if __name__ == '__main__':
#     from peft import PeftModel
#     base = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         trust_remote_code=True,
#         device_map="cpu"
#     )
#     adapter = PeftModel.from_pretrained(base, './lora_adapter_output')
#     adapter.eval()
#     text = "这家酒店的房间很舒服,但是服务人员态度很差."
#     prompt = f"评论: {text}####"
#     inputs = tokenizer(prompt, return_tensors="pt").to(adapter.device)
#     out = adapter.generate(**inputs, max_new_tokens=128)
#     result = tokenizer.decode(out[0], skip_special_tokens=True)
#     print("🔍 推理结果:", result)
