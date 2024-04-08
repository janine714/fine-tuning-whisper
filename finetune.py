from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,
                          WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer)
import torch
from dataclasses import dataclass
import jiwer

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small", cache_dir=cache_dir)
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", cache_dir=cache_dir)
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

def preprocess_function(example, text_key="text"):
    audio = example["audio"]
    labels = tokenizer(example[text_key], return_tensors="pt").input_ids.squeeze().numpy()
    input_features = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_values.squeeze().numpy()
    return {"input_features": input_features, "labels": labels}

# Load and preprocess training datasets
dataset_names_configs = [
    ("KTH/nst", "speech"), 
    ("bond005/sova_rudevices", None),
    ("conversy/liepa", None), 
    ("conversy/M-AILABS_de", None)
]

combined_datasets = []
for name, config in dataset_names_configs:
    ds = load_dataset(name, config, split="train[:20000]", cache_dir=cache_dir, download_mode="force_redownload")
    text_key = "transcription" if name == "bond005/sova_rudevices" else "text"
    text_key = "sentence" if name == "conversy/M-AILABS_de" else text_key
    processed_ds = ds.map(lambda x: preprocess_function(x, text_key=text_key), batched=True, remove_columns=ds.column_names)
    combined_datasets.append(processed_ds)
    
combined_train_datasets = concatenate_datasets(combined_datasets)

# Load and preprocess Google/Fleurs for testing
fleurs_test = load_dataset("google/fleurs", "speech_translation", split="test[:1800]", cache_dir=cache_dir)
fleurs_test_processed = fleurs_test.map(lambda x: preprocess_function(x, text_key="translation"), batched=True, remove_columns=fleurs_test.column_names)

# Load and preprocess Common Voice datasets for validation
languages = ["sv", "de", "ru", "lt"]
cv_validation_datasets = []
for lang in languages:
    cv_dataset = load_dataset("mozilla-foundation/common_voice_13_0", lang, split="train[:225]", cache_dir=cache_dir)  # 225 samples per language to total 900
    processed_cv = cv_dataset.map(lambda x: preprocess_function(x, text_key="sentence"), batched=True, remove_columns=cv_dataset.column_names)
    cv_validation_datasets.append(processed_cv)

cv_validation_combined = concatenate_datasets(cv_validation_datasets)

# Assuming the remaining dataset needs to be split for additional validation data
additional_validation_samples_needed = 900  # Adjust if needed
train_dataset, additional_validation_dataset = combined_train_datasets.train_test_split(test_size=additional_validation_samples_needed, seed=42)

# Combine CV validation data with additional validation data extracted from training data
validation_dataset = concatenate_datasets([cv_validation_combined, additional_validation_dataset])

# Prepare data collator
@dataclass
class DataCollatorWithPadding:
    processor: Any
    
    def __call__(self, features):
        input_features = [{"input_values": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)
        return {"input_values": batch["input_values"], "labels": labels}

data_collator = DataCollatorWithPadding(processor=processor)

# Define compute metrics function for evaluation
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": jiwer.wer(label_str, pred_str)}
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", cache_dir=cache_dir)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(cache_dir, "whisper_finetune_results"),
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=common_voice_concatenated,  
    eval_dataset=fleurs_processed,  
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)


trainer.train()
