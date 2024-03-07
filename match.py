from transformers import pipeline
from datasets import load_dataset
import os
import pandas as pd
import torch

cache_dir = "/proj/uppmax2024-2-2/tswa2641/huggingface"
os.environ["TRANSFORMERS_CACHE"] = cache_dir

# Load the dataset
dataset = load_dataset("KTH/nst", "speech", download_mode="reuse_dataset_if_exists", split="train[:20000]")

# Initialize ASR pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    device=-1,  # Use CPU
)

# Initialize match list
match = []
# Generate transcriptions for the audio data
predictions = []
references = []

# Access the 'text' field from the main dataset
reference_text = dataset["text"]

for audio in dataset["audio"]:
    audio_tensor = torch.tensor(audio["array"], dtype=torch.float32)  # Convert audio to tensor
    audio_array = audio_tensor.cpu().detach().numpy()  # Convert torch tensor to numpy ndarray

    result = pipe(audio_array)  # Pass numpy ndarray to the pipeline

    if result and isinstance(result, list) and result[0] and isinstance(result[0], dict) and 'text' in result[0]:
        predicted_transcription = result[0]['text']
        predictions.append(predicted_transcription)
        references.append(reference_text)  # Use the 'text' field from the main dataset
        if predicted_transcription == reference_text:
            match.append(1)  # Append 1 if prediction matches reference
        else:
            match.append(0)  # Append 0 if prediction doesn't match reference
    else:
        print("No transcription generated for this audio.")
        predictions.append(None)
        references.append(reference_text)  # Use the 'text' field from the main dataset
        match.append(0)  # Assuming no transcription means it doesn't match

# Create a Pandas DataFrame
data = {"prediction": predictions, "reference": references, "match": match}
df = pd.DataFrame(data)

# Save to CSV file
csv_path = "/proj/uppmax2024-2-2/tswa2641/whisper_transcriptions.csv"
df.to_csv(csv_path, index=False)

print(f"CSV file saved to {csv_path} successfully")



