import sounddevice as sd
import piper
import numpy as np
import torch
import re
import time


# Get device


class Mouth:
    def __init__(self, device='cpu'):
        self.model = piper.PiperVoice.load(model_path='models/en_US-ryan-high.onnx',
                                           config_path='models/en_en_US_ryan_high_en_US-ryan-high.onnx.json',
                                           use_cuda=True if device == 'cuda' else False)
        self.sample_rate = self.model.config.sample_rate

    def run_tts(self, text):
        audio = b''
        for i in self.model.synthesize_stream_raw(text):
            audio += i
        return np.frombuffer(audio, dtype=np.int16)

    def say(self, text, listen_interruption_func):
        output = self.run_tts(text)
        # get the duration of audio
        duration = len(output) / self.sample_rate
        sd.play(output, samplerate=self.sample_rate)
        interruption = listen_interruption_func(duration)
        if interruption:
            sd.stop()
            return True
        else:
            sd.wait()
            return False

    def say_multiple(self, text, listen_interruption_func):
        # split the text into sentences but keep the punctuation
        pattern = r'(?<=[.!?]) +'
        sentences = re.split(pattern, text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        # merge small sentences to make a bigger sentence
        for i in range(len(sentences) - 1):
            if len(sentences[i].split()) < 5:
                sentences[i] = sentences[i] + ' ' + sentences[i + 1]
                sentences[i + 1] = ''
        print(sentences)
        for sentence in sentences:
            if sentence == '':
                continue
            interruption = self.say(sentence, listen_interruption_func)
            time.sleep(0.025)
            if interruption:
                break


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mouth = Mouth(device=device)

    text = ("If there's one thing that makes me nervous about the future of self-driving cars, it's that they'll "
            "replace human drivers.\nI think there's a huge opportunity to make human-driven cars safer and more "
            "efficient. There's no reason why we can't combine the benefits of self-driving cars with the ease of use "
            "of human-driven cars.")
    print(text)
    mouth.say(text, lambda x: False)
    sd.wait()
