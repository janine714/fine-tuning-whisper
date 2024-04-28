import os
import pandas as pd
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset
from torch.optim.lr_scheduler import StepLR
from jiwer import wer, cer
from pathlib import Path

cache_dir = "/proj/uppmax2024-2-2/tswa2641/huggingface"
os.environ["TRANSFORMERS_CACHE"] = cache_dir

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").cuda()
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")

def prepare_audio_file(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio

def evaluate(dataset, sample_size, transcription_field, language_code):
    total_wer, total_cer = 0, 0
    model.eval()
    with torch.no_grad():
        for _, row in dataset.iterrows():
            audio = prepare_audio_file(row['file_path'])
            features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.cuda()

            predicted_ids = model.generate(features)
            hypothesis = processor.decode(predicted_ids[0], skip_special_tokens=True)
            reference = row[transcription_field]
            total_wer += wer(reference, hypothesis)
            total_cer += cer(reference, hypothesis)

    avg_wer = total_wer / sample_size
    avg_cer = total_cer / sample_size
    print(f"{language_code} Dataset - Average WER: {avg_wer:.2f}, Average CER: {avg_cer:.2f}")

def baseline_evaluation(csv_path, sample_sizes):
    df = pd.read_csv(csv_path)
    for size in sample_sizes:
        sample_df = df.sample(n=size)
        evaluate(sample_df, size, "transcription", "CSV Baseline")
def baseline_fleurs_evaluation(sample_sizes):
    languages_fleurs = ["de_de", "lt_lt", "sv_se", "pl_pl", "ru_ru"]
    for fl_lang in languages_fleurs:
        for size in sample_sizes:
            fl_dataset = load_dataset("google/fleurs", fl_lang, split=f"test[:{size}]", download_mode="reuse_dataset_if_exists")
            evaluate(fl_dataset, size, "transcription", f"Baseline FLEURS {fl_lang} for {size} samples")

def retrain_model(csv_path, sample_sizes=[10, 100, 1000]):
    baseline_evaluation(csv_path, sample_sizes) 
    baseline_fleurs_evaluation(sample_sizes) 

    df = pd.read_csv(csv_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.0)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 10
    total_steps = 0
    for size in sample_sizes:
        sample_df = df.sample(n=size)
        model.train()
        for idx, row in sample_df.iterrows():
            file_path = row['file_path']
            audio = prepare_audio_file(file_path)
            features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.cuda()
            labels = processor.tokenizer(row['transcription'], return_tensors="pt").input_ids.cuda()

            outputs = model(input_features=features, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Post-training evaluation on CSV training subset
        evaluate(sample_df, size, "transcription", f"Post-training {size} samples CSV")

        # Post-training evaluation on FLEURS for multiple languages
        languages_fleurs = ["de_de", "lt_lt", "sv_se", "pl_pl", "ru_ru"]
        for fl_lang in languages_fleurs:
            fl_dataset = load_dataset("google/fleurs", fl_lang, split=f"test[:{size}]", download_mode="reuse_dataset_if_exists")
            evaluate(fl_dataset, size, "transcription", f"FLEURS {fl_lang} Post-training")

csv_path_de = "/proj/uppmax2024-2-2/tswa2641/de_whisper_transcriptions.csv"
retrain_model(csv_path_de)


