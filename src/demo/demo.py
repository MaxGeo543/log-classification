import torch
from datasets import Dataset
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, BertModel


### Prepare Your Log Message Dataset
# Load from list (or use `load_dataset("text", data_files={"train": "logs.txt"})`)
logs = [
    "Failed to connect to DB",
    "Disk usage exceeded threshold",
    "User login failed",
    "Service started on port 8080"
]

# dataset = Dataset.from_list(logs)




### Load BERT and Tokenizer for Masked Language Modeling


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Tokenize and return PyTorch tensors
for i in range(len(logs)):
    inputs = tokenizer(
        logs[i],
        max_length=16,
        return_tensors="pt",       # Return PyTorch tensors
        padding="max_length",              # Pad to max length if batching
        truncation=True            # Truncate if too long
    )
    print(inputs)
    print(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))

quit()



### Fine tune
training_args = TrainingArguments(
    output_dir="./bert-log-mlm",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained("fine-tuned-bert-for-logs")
tokenizer.save_pretrained("fine-tuned-bert-for-logs")


### 5. Use [CLS] Token as Log Embedding
embedding_model = BertModel.from_pretrained("fine-tuned-bert-for-logs")

def embed_log_message(msg: str):
    inputs = tokenizer(msg, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (1, 768)
    return cls_embedding.squeeze(0).numpy()