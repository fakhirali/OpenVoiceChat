# Adding custom models

One of the goals of open voice chat is for it to be very easy to add new models
to the system. The code is meant to be process level not object level. We have made it so
that you can add your own models to the system with very little effort, so when the models 
improve so does the overall system.

## STT

To add a new STT model you need to create a new class that inherits from the `BaseEar` class.
This class should implement the `transcribe` method that takes in an audio np array and returns
a string.

Here is how the huggingface model is implemented:

```python
class Ear_hf(BaseEar):
    def __init__(self, model_id='openai/whisper-base.en', device='cpu',
                 silence_seconds=2, generate_kwargs=None):
        super().__init__(silence_seconds)
        self.pipe = pipeline('automatic-speech-recognition', model=model_id, device=device)
        self.device = device
        self.generate_kwargs = generate_kwargs

    @torch.no_grad()
    def transcribe(self, audio):
        transcription = self.pipe(audio, generate_kwargs=self.generate_kwargs)
        return transcription['text'].strip()
```

## TTS

To add a new TTS model you need to create a new class that inherits from the `BaseMouth` class.
This class should implement the `run_tts` method that takes in a string and returns an audio np array.

Here is how the piper-tts model is implemented:
```python
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
```

## LLM

To add a new LLM model you need to create a new class that inherits from the `BaseChatbot` class.
This class should implement the `run` method that takes in the user message and yields the assistant's response, and
the `post_process` method that takes in the response of the assistant does any post-processing and returns it.

Here is how the gpt model is implemented:
```python
class Chatbot_gpt(BaseChatbot):
    def __init__(self, sys_prompt='', Model='gpt-3.5-turbo'):
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.MODEL = Model
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.messages = []
        self.messages.append({"role": "system", "content": sys_prompt})

    def run(self, input_text):
        self.messages.append({"role": "user", "content": input_text})
        stream = self.client.chat.completions.create(
            model=self.MODEL,
            messages=self.messages,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def post_process(self, response):
        self.messages.append({"role": "assistant", "content": response})
        return response
```