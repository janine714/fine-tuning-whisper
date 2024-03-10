import os
import torch
import csv
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

cache_dir = "/proj/uppmax2024-2-2/tswa2641/huggingface"
os.environ["TRANSFORMERS_CACHE"] = cache_dir

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

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

dataset = load_dataset("KTH/nst", "speech", download_mode="reuse_dataset_if_exists", split="train[:20000]")

csv_path = "/proj/uppmax2024-2-2/tswa2641/whisper_transcriptions.csv"

# Open CSV file for writing
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = ['prediction', 'reference', 'match']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    processed_references = set()  # Keep track of processed references

    for data_item in dataset:
        reference = data_item["text"]
        if reference in processed_references:
            continue  # Skip if reference has been processed before

        sample_audio_array = data_item["audio"]["array"]

        # Choose either path or array based on your data structure
        sample_audio_data = sample_audio_array  # or sample_audio_array

        # Get the ASR result
        result = pipe(sample_audio_data, generate_kwargs={"language": "sv"})
        prediction = result['text']

        # Compare prediction with reference
        match = 1 if prediction.strip().lower() == reference.strip().lower() else 0

        # Write to CSV file
        writer.writerow({'prediction': prediction, 'reference': reference, 'match': match})

        # Mark reference as processed
        processed_references.add(reference)

        print("ASR Result:", prediction)









