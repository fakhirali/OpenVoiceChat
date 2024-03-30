import torch
from seamless_communication.models.inference import Translator as Translator_m4t
from utils import record
import sounddevice as sd



class Translator:
    def __init__(self, device, lang='urd'):
        self.model = Translator_m4t("seamlessM4T_medium", vocoder_name_or_card="vocoder_36langs",
                                    device=torch.device(device))
        self.lang = lang
    @torch.no_grad()
    def say(self, text):
        translated_text, wav, sr = self.model.predict(text, "t2st", self.lang, src_lang='eng')
        sd.play(wav.cpu().numpy()[0][0], sr)
        sd.wait()
        return translated_text
    @torch.no_grad()
    def listen(self):
        audio = record(2)
        audio = torch.tensor(audio).unsqueeze(1)
        translated_text, _, _ = self.model.predict(audio, "s2tt", 'eng')
        return str(translated_text)


if __name__ == "__main__":
    translator = Translator_m4t("seamlessM4T_medium", vocoder_name_or_card="vocoder_36langs", device=torch.device("cuda:0"))
    audio = record(3)
    audio = torch.tensor(audio).unsqueeze(1) #idk why but this is the shape
    translated_text, wav, sr = translator.predict(audio, "s2st", 'urd')
    sd.play(wav.cpu().numpy()[0][0], sr)
    sd.wait()
    print(translated_text)




