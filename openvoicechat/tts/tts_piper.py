import sounddevice as sd
import piper
import numpy as np
import torch
from tts.base import BaseMouth


# Get device


class Mouth_piper(BaseMouth):
    def __init__(self, device='cpu', model_path='models/en_US-ryan-high.onnx',
                 config_path='models/en_en_US_ryan_high_en_US-ryan-high.onnx.json'):
        self.model = piper.PiperVoice.load(model_path=model_path,
                                           config_path=config_path,
                                           use_cuda=True if device == 'cuda' else False)
        super().__init__(sample_rate=self.model.config.sample_rate)

    def run_tts(self, text):
        audio = b''
        for i in self.model.synthesize_stream_raw(text):
            audio += i
        return np.frombuffer(audio, dtype=np.int16)



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mouth = Mouth_piper(device=device, model_path='../models/en_US-ryan-high.onnx',
                        config_path='../models/en_en_US_ryan_high_en_US-ryan-high.onnx.json')

    text = ("If there's one thing that makes me nervous about the future of self-driving cars, it's that they'll "
            "replace human drivers.\nI think there's a huge opportunity to make human-driven cars safer and more "
            "efficient. There's no reason why we can't combine the benefits of self-driving cars with the ease of use "
            "of human-driven cars.")
    print(text)
    mouth.say_multiple(text, lambda x: False)
    sd.wait()
