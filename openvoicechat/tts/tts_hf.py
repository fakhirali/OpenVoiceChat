from transformers import pipeline
import sounddevice as sd
import torch
from tts.base import BaseMouth

print()
print()


class Mouth_hf(BaseMouth):
    def __init__(self, model_id='kakao-enterprise/vits-vctk', device='cpu',
                 forward_params=None):
        self.pipe = pipeline('text-to-speech', model=model_id, device=device)
        self.device = device
        self.forward_params = forward_params
        super().__init__(sample_rate=self.pipe.sampling_rate)

    @torch.no_grad()
    def run_tts(self, text):
        # inputs = self.tokenizer(text, return_tensors="pt")
        # inputs = inputs.to(self.device)
        # output = self.model(**inputs, speaker_id=self.speaker_id).waveform[0].to('cpu')
        output = self.pipe(text, forward_params=self.forward_params)
        self.sample_rate = output['sampling_rate']
        return output['audio'][0]


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mouth = Mouth_hf(model_id='kakao-enterprise/vits-vctk', device=device, forward_params={"speaker_id": 10})

    text = ("If there's one thing that makes me nervous about the future of self-driving cars, it's that they'll "
            "replace human drivers.\nI think there's a huge opportunity to make human-driven cars safer and more "
            "efficient. There's no reason why we can't combine the benefits of self-driving cars with the ease of use "
            "of human-driven cars.")
    print(text)
    mouth.say_multiple(text, lambda x: False)
    sd.wait()
