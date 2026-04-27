# step-1 imports
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import json

# step-2 json to text
json_files = [
    "Data/aad.json",
    "Data/dbms.json",
    "Data/os.json",
    "Data/da.json",
    "Data/se.json"
]

all_texts = []

for file in json_files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        text = f"""
Type: {item.get('type', '')}
Exam: {item.get('exam', '')}
Subject: {item.get('subject', '')} ({item.get('subject_code', '')})
Marks: {item.get('marks', '')}
Question: {item.get('question', '')}
Answer: {item.get('answer', '')}
"""
        all_texts.append({"text": text.strip()})

dataset = Dataset.from_list(all_texts)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset['train']
eval_dataset = dataset['test']

# step-3 bnb config + tokenizer + load model
model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map='auto'
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# step-v apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=['q_proj', 'v_proj'],
    lora_dropout=0.1,
    bias='none',
    task_type='CAUSAL_LM'
)
model = get_peft_model(model, lora_config)

print(f'Total Parameters: {sum(p.numel() for p in model.parameters()):,}')
print(f'Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n')

# step-5 prepare dataset
def format_fn(example):
    return {'text': example['text']}

train_dataset = train_dataset.map(format_fn)
eval_dataset = eval_dataset.map(format_fn)

# step-6 tokenize dataset
def tokenize_fn(example):
    tokens = tokenizer(
        example['text'],
        truncation=True,
        padding='max_length',
        max_length=512
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(tokenize_fn, batched=True, remove_columns=eval_dataset.column_names)

train_dataset.set_format('torch')
eval_dataset.set_format('torch')

# step-7 data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# step-8 training args
training_args = TrainingArguments(
    output_dir='./saved_model/tiny_llama',
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    eval_strategy='epoch',
    logging_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    lr_scheduler_type='linear',
    warmup_steps=500,
    max_grad_norm=1.0,
    fp16=torch.cuda.is_available(),
    save_total_limit=2,
    remove_unused_columns=True,
    dataloader_num_workers=0,
    report_to='none',
    run_name='epoch_loss_clean_logs',
    seed=42,
    disable_tqdm=False,
    push_to_hub=False,
    resume_from_checkpoint=None,
    optim='paged_adamw_8bit'
)

# step-9 setup trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

# step-10 train
trainer.train()

# step-11 save final model
trainer.save_model("./saved_model/tiny_llama/final_adapter")
tokenizer.save_pretrained("./saved_model/tiny_llama/final_adapter")

# step-12 clean training summary
print("\n===== TRAINING SUMMARY =====\n")
logs = trainer.state.log_history
train_losses = [(log["epoch"], log["loss"]) for log in logs if "loss" in log and "epoch" in log]
eval_losses = [(log["epoch"], log["eval_loss"]) for log in logs if "eval_loss" in log and "epoch" in log]

for epoch, train_loss in train_losses:
    eval_loss = next(v for e, v in eval_losses if e == epoch)
    print(f"Epoch {int(epoch)} → train_loss = {train_loss:.4f} | eval_loss = {eval_loss:.4f}")