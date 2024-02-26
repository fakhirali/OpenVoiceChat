<div align="center">

![logo](media/logo.gif)

<h3>

Have a natural conversation with an LLM

</h3>

</div>

---

Uses open source stt, tts and llm models.
Supports interruptions with the help of [silero VAD](https://github.com/snakers4/silero-vad).
Well [abstracted](/llm) apis, easy to use and extend.
Runs locally on a [consumer GPU](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3080-3080ti/).
[Installation](INSTALL.md).
Some ideas are [here](notes/Ideas.md)


```shell 
python main.py
```

---

### Features
- low latency
- handles interruptions
- supports multiple stt, tts and llm models from different sources


### TODO:
- [ ] OpenAI GPT support
- [ ] Streaming everything (stt, tts, llm)
- [ ] UI
- [ ] Web interface/API
- [ ] Good abstractions for streaming stt output (vosk, wav2vec2 can be streamed)

### Bug Fix:
- [ ] stt hf model should be able to tkae any models not just whisper
- [ ] Fix streaming tortoise interruption
- [ ] Better tts abstractions with multiple_say and interruption
- [ ] Super classes of stt, tts and llm for the things that are common
- [ ] Sounddevice underrun error

