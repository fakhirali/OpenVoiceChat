from tts.tts_piper import Mouth_piper as Mouth
from tts.tts_elevenlabs import Mouth_elevenlabs as Mouth
# from tts.tts_xtts import Mouth_xtts as Mouth

from llm.llm_llama import Chatbot_llama as Chatbot
# from llm.llm_gpt import Chatbot_gpt as Chatbot

# from llm import Chatbot_hf as Chatbot

from stt.stt_hf import Ear_hf as Ear
# from stt.stt_vosk import Ear_vosk as Ear

import torch
from preprompts import call_pre_prompt, llama_sales
import torchaudio
import torchaudio.functional as F
import numpy as np
import threading
import queue

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('loading models...')

    ear = Ear(device=device, silence_seconds=2)
    audio, sr = torchaudio.load('media/my_voice.wav')
    audio = F.resample(audio, sr, 16_000)[0]
    ear.transcribe(np.array(audio))

    john = Chatbot(sys_prompt=call_pre_prompt)

    mouth = Mouth()
    mouth.say_text('Good morning!')

    print("type: exit, quit or stop to end the chat")
    print("Chat started:")
    while True:
        user_input = ear.listen()
        if user_input.lower() in ["exit", "quit", "stop"]:
            break
        print(user_input)

        llm_output_queue = queue.Queue()
        interrupt_queue = queue.Queue()
        llm_thread = threading.Thread(target=john.generate_response_stream,
                                      args=(user_input, llm_output_queue, interrupt_queue))
        tts_thread = threading.Thread(target=mouth.say_multiple_stream,
                                      args=(llm_output_queue, ear.interrupt_listen, interrupt_queue))

        llm_thread.start()
        tts_thread.start()

        tts_thread.join()
        llm_thread.join()

        res = llm_output_queue.get()
        print(res)
        if '[END]' in res:
            break
