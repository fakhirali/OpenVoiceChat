import torch
from tortoise.api_fast import TextToSpeech
from tortoise.utils.audio import load_voice
import sounddevice as sd
import queue
import threading
import re


def play_audio_paralell(audio_queue, sample_rate, listen_interruption_func):
    while True:
        chunk = audio_queue.get()
        if chunk is None:
            break
        chunk = chunk.cpu().numpy()
        # wait if already playing
        sd.play(chunk, samplerate=sample_rate)
        sd.wait()


class Mouth:
    def __init__(self, device='cpu'):
        self.model = TextToSpeech(use_deepspeed=False, kv_cache=True, half=True)
        self.sample_rate = 24000
        self.voice_samples, self.conditioning_latents = load_voice('tom')

    @torch.no_grad()
    def say(self, text, listen_interruption_func):
        audio_queue = queue.Queue()
        listen_queue = queue.Queue()  # same as audio_queue for interruption
        playback_thread = threading.Thread(target=play_audio_paralell, args=(audio_queue,
                                                                             self.sample_rate,
                                                                             listen_interruption_func))
        interruption_thread = threading.Thread(target=listen_interruption_func, args=(listen_queue,))
        playback_thread.start()
        interruption_thread.start()
        audio_generator = self.model.tts_stream(text,
                                                voice_samples=self.voice_samples,
                                                conditioning_latents=self.conditioning_latents)
        for wav_chunk in audio_generator:
            audio_queue.put(wav_chunk)
            listen_queue.put('smth')
            if interruption_thread.is_alive() is False:
                audio_queue.put(None)
                sd.stop()
                print('interruption')
                playback_thread.join()
                interruption_thread.join()
                return True
        if audio_queue.empty():
            audio_queue.put(None)
            listen_queue.put(None)
            playback_thread.join()
            interruption_thread.join()
        return False

    def say_multiple(self, text, listen_interruption_func):
        pattern = r'[.?!]'
        sentences = re.split(pattern, text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        print(sentences)
        for sentence in sentences:
            interruption = self.say(sentence, listen_interruption_func)
            if interruption:
                break


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mouth = Mouth()

    text = "If there's one thing that makes me nervous about the future of self-driving cars, it's that they'll replace human drivers.\nI think there's a huge opportunity to make human-driven cars safer and more efficient. There's no reason why we can't combine the benefits of self-driving cars with the ease of use of human-driven cars."
    print(text)
    mouth.say(text, lambda x: False)
    sd.wait()
