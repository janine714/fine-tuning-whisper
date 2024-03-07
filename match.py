import os
import pandas as pd
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

cache_dir = "/proj/uppmax2024-2-2/tswa2641/huggingface"
os.environ["TRANSFORMERS_CACHE"] = cache_dir
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


model_id = "openai/whisper-large-v3"
torch_dtype = torch.float32  
device = "cuda" 

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

# Initialize the processor
processor = AutoProcessor.from_pretrained(model_id)

# Create the ASR pipeline
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

# Load the dataset
dataset = load_dataset("KTH/nst", "speech", download_mode="reuse_dataset_if_exists", split="train[:20000]")

# Initialize match list
match = []

# Generate transcriptions for the audio data
predictions = []
references = []

for audio in dataset["audio"]:
    result = pipe(audio)
    if result:
        predicted_transcription = result[0]['text']
        predictions.append(predicted_transcription)
        references.append(audio["text"])
        if predicted_transcription == audio["text"]:
            match.append(1)  # Append 1 if prediction matches reference
        else:
            match.append(0)  # Append 0 if prediction doesn't match reference
    else:
        print("No transcription generated for this audio.")
        predictions.append(None)
        references.append(audio["text"])
        match.append(0)  # Assuming no transcription means it doesn't match

# Create a Pandas DataFrame
data = {"prediction": predictions, "reference": references, "match": match}
df = pd.DataFrame(data)

# Save to CSV file
csv_path = "/proj/uppmax2024-2-2/tswa2641/whisper_transcriptions.csv"
df.to_csv(csv_path, index=False)

print(f"CSV file saved to {csv_path} successfully")
