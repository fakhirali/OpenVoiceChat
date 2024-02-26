import torch
import numpy as np
from utils import record_user

class VoiceActivityDetection:
    def __init__(self, sampling_rate=16000):
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                model='silero_vad',
                                                force_reload=False)

        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = utils

        self.sampling_rate = sampling_rate

    def contains_speech(self, audio):
        frames = np.frombuffer(b''.join(audio), dtype=np.int16)

        # normalization see https://discuss.pytorch.org/t/torchaudio-load-normalization-question/71470
        frames = frames / (1 << 15)

        audio = torch.tensor(frames.astype(np.float32))
        speech_timestamps = self.get_speech_timestamps(audio, self.model,
                                                       sampling_rate=self.sampling_rate)
        return len(speech_timestamps) > 0


if __name__ == "__main__":
    vad = VoiceActivityDetection()
    audio = record_user(3, vad)



