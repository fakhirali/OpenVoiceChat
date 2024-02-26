from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torchaudio.functional as F
import torch
from utils import record_user, record_interruption, record_interruption_parallel
from vad import VoiceActivityDetection
import re

print()
print()


class Ear:
    def __init__(self, model_id='openai/whisper-base.en', device='cpu', silence_seconds=2):
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        self.device = device
        self.model.to(device)
        self.vad = VoiceActivityDetection()
        self.silence_seconds = silence_seconds
        self.not_interrupt_words = ['you', 'yes', 'yeah', 'hmm']

    @torch.no_grad()
    def transcribe(self, audio):
        input_features = self.processor(audio, sampling_rate=16_000, return_tensors="pt").input_features.to(self.device)
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return ' '.join(transcription)

    def listen(self):
        audio = record_user(self.silence_seconds, self.vad)
        text = self.transcribe(audio)
        return text

    def interrupt_listen(self, record_seconds=100):
        while record_seconds > 0:
            interruption_audio = record_interruption(self.vad, record_seconds)
            # duration of interruption audio
            if interruption_audio is None:
                return False
            else:
                duration = len(interruption_audio) / 16_000
                text = self.transcribe(interruption_audio)
                # remove any punctuation using re
                text = re.sub(r'[^\w\s]', '', text)
                text = text.lower()
                text = text.strip()
                print(text)
                if text in self.not_interrupt_words:
                    record_seconds -= duration
                else:
                    return True

    def interrupt_listen_parallel(self, listen_queue):
        while True:
            interruption_audio = record_interruption_parallel(self.vad, listen_queue)
            # duration of interruption audio
            if interruption_audio is None:
                return False
            else:
                text = self.transcribe(interruption_audio)
                # remove any punctuation using re
                text = re.sub(r'[^\w\s]', '', text)
                text = text.lower()
                text = text.strip()
                print(text)
                if text not in self.not_interrupt_words:
                    return True


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ear = Ear(device=device)

    audio, sr = torchaudio.load('media/abs.wav')
    audio = F.resample(audio, sr, 16_000)[0]
    text = ear.transcribe(audio)
    print(text)

    text = ear.listen()
    print(text)
