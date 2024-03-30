from TTS.api import TTS
import sounddevice as sd
import torch
from tts.base import BaseMouth


# Get device


class Mouth_xtts(BaseMouth):
    def __init__(self, model_id='tts_models/en/jenny/jenny', device='cpu'):
        self.model = TTS(model_id)
        self.device = device
        self.model.to(device)
        super().__init__(sample_rate=self.model.synthesizer.output_sample_rate)

    def run_tts(self, text):
        output = self.model.tts(text=text, split_sentences=False)
        return output

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mouth = Mouth_xtts(device=device)

    text = ("If there's one thing that makes me nervous about the future of self-driving cars, it's that they'll "
            "replace human drivers.\nI think there's a huge opportunity to make human-driven cars safer and more "
            "efficient. There's no reason why we can't combine the benefits of self-driving cars with the ease of use "
            "of human-driven cars.")
    print(text)
    mouth.say_multiple(text, lambda x: False)
    sd.wait()
