import numpy as np
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperProcessor
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor
import torch
from datasets import load_dataset
from datasets import Dataset
import os
from torch.nn.utils.rnn import pad_sequence

cache_dir = "/proj/uppmax2024-2-2/tswa2641/huggingface"
os.environ["TRANSFORMERS_CACHE"] = cache_dir

model = WhisperForConditionalGeneration.from_pretrained("/proj/uppmax2024-2-2/tswa2641/results/whisper-small$
tokenizer = WhisperTokenizer.from_pretrained("/proj/uppmax2024-2-2/tswa2641/results/whisper-small-sv")
feature_extractor = WhisperFeatureExtractor.from_pretrained("/proj/uppmax2024-2-2/tswa2641/results/whisper-s$
processor = WhisperProcessor.from_pretrained("/proj/uppmax2024-2-2/tswa2641/results/whisper-small-sv")


dataset = load_dataset("KTH/nst", "speech", download_mode="reuse_dataset_if_exists", split="train[:10]")

def preprocess_function(example):
    audio_data = example["audio"]["array"]
    inputs = processor(audio_data, sampling_rate=16_000, return_tensors="pt", padding="max_length", max_length=3000, truncation=True)
    input_features = inputs.input_features[0]

    # Pad the input features if its length is less than 3000
    if input_features.shape[1] < 3000:
        padding_length = 3000 - input_features.shape[1]
        input_features = torch.nn.functional.pad(input_features, (0, padding_length), value=0)

    # Generate attention mask
    attention_mask = torch.ones_like(input_features)  # Assuming all input tokens are attended to
    input_features = torch.tensor(input_features)
    attention_mask = torch.tensor(attention_mask)

    # Print the shape and length of the input features
    print("Shape of input features:", input_features.shape)
    print("Length of input features:", input_features.shape[1])

    example["input_features"] = input_features
    example["labels"] = torch.tensor(tokenizer.encode(example["text"]))  # Convert to tensor

    # Adjust the shape of attention_mask to match input_features
    example["attention_mask"] = attention_mask.unsqueeze(1).expand(-1, input_features.shape[1], -1)
    return example


dataset = dataset.map(preprocess_function)


def collate_fn(batch):
    input_features = [torch.tensor(example["input_features"]) for example in batch]
    labels = [example["labels"] for example in batch]
    attention_masks = [torch.tensor(example["attention_mask"]) for example in batch]

    # Pad input features and attention masks to have the same length
    max_input_length = 3000
    padded_inputs = torch.nn.utils.rnn.pad_sequence(input_features, batch_first=True, padding_value=0)
    padded_attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

    # Ensure that the padded inputs and attention masks have the desired length
    if padded_inputs.shape[1] < max_input_length:
        padding_length = max_input_length - padded_inputs.shape[1]
        padded_inputs = torch.nn.functional.pad(padded_inputs, (0, padding_length), value=0)
        padded_attention_masks = torch.nn.functional.pad(padded_attention_masks, (0, padding_length), value=0)

    # Convert labels to tensor and pad to the maximum length
    max_label_length = max(len(label) for label in labels)
    padded_labels = [torch.nn.functional.pad(torch.tensor(label), (0, max_label_length - len(label))) for label in labels]
    padded_labels = torch.stack(padded_labels)

    return {"input_features": padded_inputs, "labels": padded_labels, "attention_mask": padded_attention_masks}


data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

# Iterate over DataLoader and perform inference
error_dataset = []
for batch in data_loader:
    inputs, labels, attention_masks = batch["input_features"], batch["labels"], batch["attention_mask"]

    # Print the shape and length of input features
    print("Shape of inputs batch:", inputs.shape)
    print("Length of inputs batch:", inputs.shape[1])

    predictions = model.generate(inputs, attention_mask=attention_masks)

    # Compare predictions with ground truth labels
    for prediction, label in zip(predictions, labels):
        # Decode predictions and labels
        decoded_prediction = tokenizer.decode(prediction, skip_special_tokens=True)
        decoded_label = tokenizer.decode(label, skip_special_tokens=True)

        # Check if prediction matches the ground truth label
        if decoded_prediction != decoded_label:
            # Store the error example along with the ground truth label
            error_dataset.append({"prediction": decoded_prediction, "ground_truth": decoded_label})


torch.save(error_dataset, "/proj/uppmax2024-2-2/tswa2641/swedish_error_dataset2")




