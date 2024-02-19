from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import jiwer
import os

cache_dir = "/proj/uppmax2024-2-2/tswa2641/huggingface"
os.environ["TRANSFORMERS_CACHE"] = cache_dir


ds = DatasetDict()
ds['train'] = load_dataset("KTH/nst", "speech", split="train[:50%]")

for example in ds["train"][:5]:
    print(example)
print(ds["train"].features)


for example in ds['train']:
    audio_keys = example['audio'].keys()
    break  # Break after inspecting the first example

print("Keys in 'audio' field:", audio_keys)



train_test_valid = ds['train'].train_test_split(test_size=0.2)
test_valid = train_test_valid['test'].train_test_split(test_size=0.5)

def prepare_dataset(batch):
    if "audio" not in batch or "text" not in batch:
        return {}

    # Process audio data
    audio_features = feature_extractor(batch["audio"]["array"], sampling_rate=batch["audio"]["sampling_rate"])
    if not audio_features.input_features:
        return {}
    batch["input_features"] = audio_features.input_features[0]

    # Process text data and assign input IDs directly to batch["labels"]
    text_input_ids = tokenizer(batch["text"]).input_ids
    if not text_input_ids:
        return {}
    batch["labels"] = text_input_ids
    return batch


# Define data processing components
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Swedish", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Swedish", task="transcribe")

# Map the prepare_dataset function to the dataset
ds = DatasetDict({
    'train': train_test_valid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']
})

max_train_samples = 5000
max_test_valid_samples = 1000

ds["train"] = ds["train"].select(range(max_train_samples))

# Select a subset of examples for testing and validation
ds["test"] = ds["test"].select(range(max_test_valid_samples))
ds["valid"] = ds["valid"].select(range(max_test_valid_samples))


ds = ds.map(prepare_dataset, num_proc=4, batch_size=4)


print("Train dataset size:", len(ds['train']))
print("Test dataset size:", len(ds['test']))
print("Validation dataset size:", len(ds['valid']))

# Define batch collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Define evaluation metrics
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * jiwer.wer(label_str, pred_str)
    return {"wer": wer}

# Load pre-trained model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Training Configuration
training_args = Seq2SeqTrainingArguments(
    output_dir="/proj/uppmax2024-2-2/tswa2641/results/whisper-small-sv",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

# Define trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=ds['train'],
    eval_dataset=ds['test'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)

trainer.train()

