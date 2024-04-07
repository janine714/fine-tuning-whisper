import os
import pandas as pd
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from pathlib import Path
from jiwer import wer, cer 

cache_dir = "/proj/uppmax2024-2-2/tswa2641/huggingface"
os.environ["TRANSFORMERS_CACHE"] = cache_dir

# Model and processor setup
base_dir = "/proj/uppmax2024-2-2/tswa2641/results/whisper-small-multi"
model_checkpoint = f"{base_dir}/checkpoint-4000"
model = WhisperForConditionalGeneration.from_pretrained(model_checkpoint).cuda()
processor = WhisperProcessor.from_pretrained(base_dir)

def prepare_audio_file(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio

def retrain_model(csv_path, sample_sizes=[100, 500, 1000], cer_threshold=0.2):
    df = pd.read_csv(csv_path)
    # Filter samples where CER > 0.2
    filtered_df = df[df['cer'] > cer_threshold]

    for size in sample_sizes:
        # Randomly sample error-prone examples
        sample_df = filtered_df.sample(n=size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

        model.train()
        total_wer = 0
        total_cer = 0
        print(f"\nTraining with {size} samples:")
        for idx, row in sample_df.iterrows():
            file_path = row['file_path']
            audio = prepare_audio_file(file_path)
            features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.cuda()
            with torch.no_grad():
                labels = processor.tokenizer(row['reference'], return_tensors="pt").input_ids.cuda()

            outputs = model(input_features=features, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate WER and CER
            predicted_ids = torch.argmax(outputs.logits, dim=-1
            hypothesis = processor.decode(predicted_ids[0], skip_special_tokens=True)
            reference = row['reference']
            total_wer += wer(reference, hypothesis)
            total_cer += cer(reference, hypothesis)

            # Print a prediction-reference pair for each sample size
            if idx == 0:  # print only the first pair for each sample size
                print(f"Sample 1 for size {size}:")
                print(f"Reference: {reference}")
                print(f"Prediction: {hypothesis}")

        avg_wer = total_wer / size
        avg_cer = total_cer / size
        print(f"Average WER for size {size}: {avg_wer}")
        print(f"Average CER for size {size}: {avg_cer}")

        retrained_model_path = Path(model_checkpoint).parent / f"retrained_{Path(csv_path).stem}_{size}"
        model.save_pretrained(retrained_model_path)
        processor.save_pretrained(retrained_model_path)


csv_path_de = "/proj/uppmax2024-2-2/tswa2641/de_whisper_transcriptions.csv"
retrain_model(csv_path_de)

