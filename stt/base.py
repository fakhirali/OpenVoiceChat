
import torch
from utils import record_user, record_interruption
from vad import VoiceActivityDetection
import re
from time import monotonic



class BaseEar:
    def __init__(self, silence_seconds=3, not_interrupt_words=None):
        if not_interrupt_words is None:
            not_interrupt_words = ['you', 'yes', 'yeah', 'hmm'] # you because whisper says "you" in silence
        self.silence_seconds = silence_seconds
        self.not_interrupt_words = not_interrupt_words
        self.vad = VoiceActivityDetection()

    @torch.no_grad()
    def transcribe(self, input):
        raise NotImplementedError("This method should be implemented by the subclass")

    def listen(self):
        audio = record_user(self.silence_seconds, self.vad)
        text = self.transcribe(audio)
        return text

    def listen_timing(self):
        audio = record_user(self.silence_seconds, self.vad)
        start = monotonic()
        text = self.transcribe(audio)
        end = monotonic()
        return text, end - start


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
