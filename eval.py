import librosa
from huggingface_hub import HfApi, HfFolder
import os
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments,  WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor
from datasets import load_dataset, DatasetDict
import jiwer
from dataclasses import dataclass
from typing import Any, Dict, List, Union

cache_dir = "/proj/uppmax2024-2-2/tswa2641/huggingface"
os.environ["TRANSFORMERS_CACHE"] = cache_dir

model = WhisperForConditionalGeneration.from_pretrained("/proj/uppmax2024-2-2/tswa2641/results/whisper-small-sv/checkpoint-4000")
tokenizer = WhisperTokenizer.from_pretrained("/proj/uppmax2024-2-2/tswa2641/results/whisper-small-sv")
feature_extractor = WhisperFeatureExtractor.from_pretrained("/proj/uppmax2024-2-2/tswa2641/results/whisper-small-sv")
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def prepare_dataset_swedish(batch):
    try:
        if "audio" not in batch or "text" not in batch:
            return {"input_features": [], "labels": []}

        # Resample the audio data to 16000 Hz
        audio_data = librosa.resample(batch["audio"]["array"], orig_sr=batch["audio"]["sampling_rate"], target_sr=16000)

        audio_features = feature_extractor(audio_data, sampling_rate=16000)
        if not audio_features.input_features:
            return {"input_features": [], "labels": []}
        batch["input_features"] = audio_features.input_features[0]

        text_input_ids = tokenizer(batch["text"]).input_ids
        if not text_input_ids:
            return {"input_features": [], "labels": []}
        batch["labels"] = text_input_ids
        return batch
    except Exception as e:
        print(f"Error processing batch: {e}")
        return {"input_features": [], "labels": []}

def prepare_dataset_german(batch):
    try:
        if "audio" not in batch or "sentence" not in batch:
            return {"input_features": [], "labels": []}


        audio_data = librosa.resample(batch["audio"]["array"], orig_sr=batch["audio"]["sampling_rate"], target_sr=16000)

        audio_features = feature_extractor(audio_data, sampling_rate=16000)
        if not audio_features.input_features:
            return {"input_features": [], "labels": []}
        batch["input_features"] = audio_features.input_features[0]

        text_input_ids = tokenizer(batch["sentence"]).input_ids
        if not text_input_ids:
            return {"input_features": [], "labels": []}
        batch["labels"] = text_input_ids
        return batch
    except Exception as e:
        print(f"Error processing batch: {e}")
        return {"input_features": [], "labels": []}

ds = DatasetDict()
swedish_data = load_dataset("KTH/nst", "speech", download_mode="reuse_dataset_if_exists", split="train[:10%]")
swedish_train_test = swedish_data.train_test_split(test_size=0.2)

ds['swedish_eval'] = swedish_train_test['test']
ds['swedish_eval'] = ds['swedish_eval'].map(prepare_dataset_swedish, num_proc=1, batch_size=16)

HfFolder.save_token("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")
ds['german_eval'] = load_dataset("mozilla-foundation/common_voice_13_0", "de", split="test[:50%]")
ds['german_eval'] = ds['german_eval'].map(prepare_dataset_german, num_proc=1, batch_size=16)

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
# Initialize a counter outside the function
transcription_counter = 0

def compute_metrics(pred):
    global transcription_counter  # Use the global counter
    if transcription_counter < 10:  # Only print for the first 10 batches
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Print the model's transcription and the actual transcription
        print("Model's Transcription: ", pred_str)
        print("Actual Transcription: ", label_str)

        transcription_counter += 1  # Increment the counter

    wer = 100 * jiwer.wer(label_str, pred_str)
    return {"wer": wer}


training_args = Seq2SeqTrainingArguments(
    output_dir="/proj/uppmax2024-2-2/tswa2641/results/whisper-small-sv",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    evaluation_strategy="steps",
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    eval_dataset=ds['swedish_eval'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)


eval_results = trainer.evaluate()
print(eval_results)

trainer.eval_dataset = ds['german_eval']
eval_results = trainer.evaluate()
print(eval_results)


