from tts import Mouth_piper as Mouth
# from tts import Mouth_tortoise as Mouth
# from tts import Mouth_xtts as Mouth

from llm import Chatbot_llama as Chatbot
# from llm import Chatbot_hf as Chatbot

from stt import Ear_hf as Ear
# from stt import Ear_vosk as Ear

import torch
from preprompts import call_pre_prompt
import torchaudio
import torchaudio.functional as F

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('loading models...')

    ear = Ear(device=device, silence_seconds=2)
    audio, sr = torchaudio.load('media/my_voice.wav')
    audio = F.resample(audio, sr, 16_000)[0]
    ear.transcribe(audio)

    john = Chatbot(device=device, sys_prompt=call_pre_prompt)
    # john.generate_response('hello', c)

    mouth = Mouth(device=device)
    # mouth.say('Good morning! Thank you for calling Apple. My name is John, how can I assist you today?')
    mouth.say('Good morning!', ear.interrupt_listen)

    print("type: exit, quit or stop to end the chat")
    print("Chat started:")
    while True:
        user_input = ear.listen()
        if user_input.lower() in ["exit", "quit", "stop"]:
            break
        print(user_input)
        response = john.generate_response(user_input)
        print(response)

        # mouth.say(response.replace('[USER]', '').replace('[END]', '').replace('[START]', ''), ear.interrupt_listen)
        mouth.say_multiple(response.replace('[USER]', '').replace('[END]', '').replace('[START]', ''),
                           ear.interrupt_listen)
        if response.find('[END]') != -1:
            break
