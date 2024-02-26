import vosk
import numpy as np
import json
from utils import record_user, record_interruption
import re
from vad import VoiceActivityDetection
import torchaudio
import torchaudio.functional as F
import torch


class Ear:
    def __init__(self, model_path='models/vosk-model-en-us-0.22', device='cpu', silence_seconds=2):
        self.model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        self.device = device
        self.silence_seconds = silence_seconds
        self.vad = VoiceActivityDetection()
        self.not_interrupt_words = ['you', 'yes', 'yeah', 'hmm']

    def transcribe(self, audio):
        # if audio is a tensor convert it to numpy array
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        audio = audio.astype(np.float64) * (1 << 15)
        audio = audio.astype(np.int16).tobytes()
        result_text = ''
        last_partial_text = ''
        i = 0
        while True:
            data = audio[i * 12000: (i + 1) * 12000]
            i += 1
            if len(data) == 0:
                break
            if self.recognizer.AcceptWaveform(data):
                if result_text == '':
                    result_text = json.loads(self.recognizer.Result())['text']
                else:
                    result_text += json.loads(self.recognizer.Result())['text']
            else:
                # print(self.recognizer.PartialResult())
                pass


        return result_text

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


if __name__ == "__main__":
    ear = Ear(model_path='../models/vosk-model-en-us-0.22')

    audio, sr = torchaudio.load('../media/abs.wav')
    audio = F.resample(audio, sr, 16_000)[0]
    text = ear.transcribe(np.array(audio))
    print(text)

    text = ear.listen()
    print(text)
