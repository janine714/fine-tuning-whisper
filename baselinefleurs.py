import os
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset
from jiwer import wer, cer

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").cuda()
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")

def prepare_audio_file(audio, sr, target_sr=16000):
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio

def evaluate(dataset, transcription_field, language_code):
    total_wer, total_cer = 0, 0
    model.eval()
    with torch.no_grad():
        for data_item in dataset:
            audio_data = data_item["audio"]["array"]
            sampling_rate = data_item["audio"]["sampling_rate"]
            audio = prepare_audio_file(audio_data, sampling_rate)
            features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.cuda()
            predicted_ids = model.generate(features, language=None)
            hypothesis = processor.decode(predicted_ids[0], skip_special_tokens=True)
            reference = data_item[transcription_field]
            total_wer += wer(reference, hypothesis)
            total_cer += cer(reference, hypothesis)

    avg_wer = total_wer / len(dataset)
    avg_cer = total_cer / len(dataset)
    print(f"{language_code} Dataset - Average WER: {avg_wer:.2f}, Average CER: {avg_cer:.2f}")

languages_fleurs = ["de_de", "lt_lt", "sv_se", "pl_pl", "ru_ru"]
for fl_lang in languages_fleurs:
    fl_dataset = load_dataset("google/fleurs", fl_lang, split="test", download_mode="reuse_dataset_if_exists")
    evaluate(fl_dataset, "transcription", f"FLEURS {fl_lang}")
