import os
import torch
import csv
import string
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from jiwer import cer

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-medium"

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)
dataset = load_dataset("KTH/nst", "speech", download_mode="reuse_dataset_if_exists", split="train[:2000]")
csv_path = "/proj/uppmax2024-2-2/tswa2641/swedish_whisper_transcriptions.csv"
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = ['file_path', 'prediction', 'reference', 'match', 'cer']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for data_item in dataset:
        reference = data_item["text"]
        sample_audio_path = data_item["audio"]["path"]

        # Process the audio data
        sample_audio_array = data_item["audio"]["array"]

        # Get the ASR result
        result = pipe(sample_audio_array, generate_kwargs={"language": "sv"})
        prediction = result['text']

        # Normalize punctuation for both prediction and reference
        prediction_normalized = prediction.translate(str.maketrans('', '', string.punctuation)).lower()
        reference_normalized = reference.translate(str.maketrans('', '', string.punctuation)).lower()

        # Compare prediction with reference
        match = 1 if prediction_normalized == reference_normalized else 0

        # Calculate Character Error Rate
        cerscore = cer(reference_normalized, prediction_normalized)

        # Write to CSV file, including the file path
        writer.writerow({
            'file_path': sample_audio_path,
            'prediction': prediction,
            'reference': reference,
            'match': match,
            'cer': cerscore
        })

        print("Processed:", sample_audio_path)







