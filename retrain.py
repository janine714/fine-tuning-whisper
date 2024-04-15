import os
import pandas as pd
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset
from pathlib import Path
from torch.optim.lr_scheduler import StepLR
from jiwer import wer, cer


cache_dir = "/proj/uppmax2024-2-2/tswa2641/huggingface"
os.environ["TRANSFORMERS_CACHE"] = cache_dir

model_checkpoint = "openai/whisper-large"
model = WhisperForConditionalGeneration.from_pretrained(model_checkpoint).cuda()
processor = WhisperProcessor.from_pretrained(model_checkpoint)

def prepare_audio_file(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio

def evaluate_on_fleurs(model, processor, languages, sample_size):
    for language_code in languages:
        dataset = load_dataset("google/fleurs", language_code, split="test[:5000]", download_mode="reuse_dataset_if_exists")
        subset = dataset.shuffle(seed=42).select(range(sample_size))
        total_wer, total_cer = 0, 0

        model.eval()
        with torch.no_grad():
            for data_item in subset:
                audio_data = data_item["audio"]["array"]
                sampling_rate = data_item["audio"]["sampling_rate"]
                audio = prepare_audio_file(audio_data, sampling_rate)
                features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.cuda()

                predicted_ids = model.generate(features)
                hypothesis = processor.decode(predicted_ids[0], skip_special_tokens=True)
                reference = data_item["transcription"]
                total_wer += wer(reference, hypothesis)
                total_cer += cer(reference, hypothesis)

        avg_wer = total_wer / sample_size
        avg_cer = total_cer / sample_size
        print(f"FLEURS - Language: {language_code} - Sample Size: {sample_size} - Average WER: {avg_wer}, Average CER: {avg_cer}")

def evaluate_on_common_voice(model, processor, languages, sample_size):
    for language_code in languages:
        dataset_name = "mozilla-foundation/common_voice_13_0"
        try:
            dataset = load_dataset(dataset_name, language_code, split=f"test[:5000]", download_mode="reuse_dataset_if_exists")
        except Exception as e:
            print(f"Skipping {language_code} for {dataset_name} due to error: {e}")
            continue
        subset = dataset.shuffle(seed=42).select(range(sample_size))
        total_wer, total_cer = 0, 0

        model.eval()
        with torch.no_grad():
            for data_item in subset:
                audio_data = data_item["audio"]["array"]
                sampling_rate = data_item["audio"]["sampling_rate"]
                audio = prepare_audio_file(audio_data, sampling_rate)
                features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.cuda()

                predicted_ids = model.generate(features)
                hypothesis = processor.decode(predicted_ids[0], skip_special_tokens=True)
                reference = data_item["sentence"]
                total_wer += wer(reference, hypothesis)
                total_cer += cer(reference, hypothesis)

        avg_wer = total_wer / sample_size
        avg_cer = total_cer / sample_size
        print(f"{dataset_name.upper()} - Language: {language_code} - Sample Size: {sample_size} - Average WER: {avg_wer}, Average CER: {avg_cer}")


def train_and_evaluate(csv_path, sample_sizes=[100, 500, 1000]):
    df = pd.read_csv(csv_path)
    common_voice_languages = ["de", "sv-SE", "lt", "pl", "ru"]  
    fleurs_languages = ["de_de", "sv_se", "lt_lt", "pl_pl", "ru_ru"]  

    for size in sample_sizes:
        sample_df = df.sample(n=size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.03)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        
        model.train()
        for idx, row in sample_df.iterrows():
            audio = prepare_audio_file(row['file_path'])
            features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.cuda()
            labels = processor.tokenizer(row['reference'], return_tensors="pt").input_ids.cuda()

            outputs = model(input_features=features, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        
        evaluate_on_fleurs(model, processor, fleurs_languages, size)
        evaluate_on_common_voice(model, processor, common_voice_languages, size)


csv_path_de = "/proj/uppmax2024-2-2/tswa2641/de_whisper_transcriptions.csv"
train_and_evaluate(csv_path_de)

