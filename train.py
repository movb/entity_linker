import json
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, TrainingArguments, DataCollatorWithPadding, Trainer
from datasets import Dataset


from .config import config

# Load the JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

file_path = config['data']['file_path']
data = load_json_data(file_path)

# Preprocess the data
def preprocess_data(data):
    processed_data = []
    for page in data:
        for mention in page['mentions']:
            processed_data.append({
                'input_text': mention['text'],
                'entity': mention['entity'],
                'entity_text': page['text']
            })
    return processed_data

processed_data = preprocess_data(data)
train_data, test_data = train_test_split(processed_data, test_size=0.2, random_state=42)

train_dataset = Dataset.from_dict(train_data)
test_dataset = Dataset.from_dict(test_data)

# Train the bi-encoder model
text_model_name = config['model']['text_model_name']
entity_model_name = config['model']['entity_model_name']
tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_encoder = AutoModel.from_pretrained(text_model_name)
entity_encoder = AutoModel.from_pretrained(entity_model_name)


def tokenize_data(example):
    input_text = example['input_text']
    entity_text = example['entity_text']
    input_encoding = tokenizer(input_text, truncation=True, padding=False)
    entity_encoding = tokenizer(entity_text, truncation=True, padding=False)
    return {**input_encoding, 'entity_ids': entity_encoding['input_ids'], 'entity_attention_mask': entity_encoding['attention_mask']}

train_dataset = train_dataset.map(tokenize_data, batched=True)
test_dataset = test_dataset.map(tokenize_data, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="bi_encoder_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    seed=42,
    learning_rate=5e-5,
)

def compute_loss(text_encoder, entity_encoder, inputs, return_outputs=False):
    input_ids = inputs.pop("input_ids")
    attention_mask = inputs.pop("attention_mask")
    entity_ids = inputs.pop("entity_ids")
    entity_attention_mask = inputs.pop("entity_attention_mask")

    input_outputs = text_encoder(input_ids, attention_mask=attention_mask)
    entity_outputs = entity_encoder(entity_ids, attention_mask=entity_attention_mask)

    similarities = (input_outputs[0] * entity_outputs[0]).sum(dim=1)
    loss = -similarities.mean()

    if return_outputs:
        return loss, input_outputs
    return loss

trainer = Trainer(
    model=(text_encoder, entity_encoder),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_loss=compute_loss,
)

trainer.train()

# Save and test the model
text_encoder.save_pretrained(config['model']['text_encoder_path'])
entity_encoder.save_pretrained(config['model']['entity_encoder_path'])
tokenizer.save_pretrained(config['model']['tokenizer_path'])