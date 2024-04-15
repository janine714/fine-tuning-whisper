from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,
                          WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer)
import torch
from dataclasses import dataclass
import os
import jiwer


cache_dir = "/proj/uppmax2024-2-2/tswa2641/huggingface"
os.environ["TRANSFORMERS_CACHE"] = cache_dir

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large", cache_dir=cache_dir)
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large", cache_dir=cache_dir)
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

def preprocess_function(example, text_key="text"):
    audio = example["audio"]
    labels = tokenizer(example[text_key], return_tensors="pt").input_ids.squeeze().numpy()
    input_features = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_values.squeeze().numpy()
    return {"input_features": input_features, "labels": labels}

dataset_names_configs = [
    ("KTH/nst", "speech"), 
    ("bond005/sova_rudevices", None),
    ("conversy/liepa", None), 
    ("conversy/M-AILABS_de", None),
    ("conversy/M-AILABS_pl", None)  
]

combined_datasets = []
for name, config in dataset_names_configs:
    ds = load_dataset(name, config, split="train[:20000]", cache_dir=cache_dir, download_mode="force_redownload")
    text_key = "transcription" if name == "bond005/sova_rudevices" else "text"
    text_key = "sentence" if name in ["conversy/M-AILABS_de", "conversy/M-AILABS_pl"] else text_key
    processed_ds = ds.map(lambda x: preprocess_function(x, text_key=text_key), batched=True, remove_columns=ds.column_names)
    combined_datasets.append(processed_ds)
    
combined_train_datasets = concatenate_datasets(combined_datasets)
combined_train_datasets = combined_train_datasets.shuffle(seed=42)


languages = ["sv", "de", "ru", "lt”, “pl”]
cv_validation_datasets = []
for lang in languages:
    cv_dataset = load_dataset("mozilla-foundation/common_voice_13_0", lang, split="validation[:225]", cache_dir=cache_dir)  # Adjusted to validation split
    processed_cv = cv_dataset.map(lambda x: preprocess_function(x, text_key="sentence"), batched=True, remove_columns=cv_dataset.column_names)
    cv_validation_datasets.append(processed_cv)

cv_validation_combined = concatenate_datasets(cv_validation_datasets)

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

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large", cache_dir=cache_dir)

training_args = Seq2SeqTrainingArguments(
    output_dir="/proj/uppmax2024-2-2/tswa2641/results/whisper-large-multi",
    per_device_train_batch_size=2,  
    gradient_accumulation_steps=8,  
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=1, 
    max_steps=4000,  
    gradient_checkpointing=False,
    fp16=True,
    evaluation_strategy="no",
    per_device_eval_batch_size=2,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    push_to_hub=False,
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=combined_train_datasets,  
    eval_dataset=cv_validation_combined,  # Use only Common Voice for validation
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()
