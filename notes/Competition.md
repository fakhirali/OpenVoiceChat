# Other Voice Chat solutions on the market

## Commercial

### [elto.ai](https://elto.ai)

The tts sounds very clean. There is a delay before the
response. Probably just making sure that the person is done 
speaking. Also, if the LLM output is computed before, this 
may also be a reason for the delay. 
They didn't show any interruptions.
They say it is for standard use cases, the chatbot could just
be a RASA bot.


### [Air.ai](https://air.ai)

The tts is not that great. They're probably using 
vits, the voice dies down at the end. The response time
is alright. They support interruptions. The chatbot is 
an LLM. 

### [NLPearl](https://nlpearl.ai/)

[Demo](https://www.youtube.com/watch?v=VX7bAvkQkNQ&t=92s&ab_channel=PEARL)
Good response time. The voice is alright. Does do interruptions. 
Does a "umm" when interrupted. Not sure whether the chatbot is an LLM.

### [Bland ai](https://bland.ai)

Is alright. The sound is good, I think it also does interruptions.

### [Retell AI](https://www.retellai.com/)

Same thing. Everyone is doing it. They give some info about 
what they are doing [here](https://docs.retellai.com/blog/build-voice-agent)
They have a turn taking model.

### [SmartChild](https://smarterchild.chat/)

This it the best one so far. Very very fast.
But interruptions are not great. It speaks over me a lot.

### [Call Annie](https://callannie.ai/)

Good response time, no interruptions tho. 
The Avatar is very responsive. Seems to be able to do 
function calling.

### [Vapi AI](https://vapi.ai/)

Similar to Retell AI, they also have an endpointing model.

## Open Source

### [Bud-e](https://github.com/LAION-AI/natural_voice_assistant)

A project by LAION AI. The code base is not very clean.
Seems like a standalone project. Very talented people working.
Good discussion and research on their discord.
Interruptions aren't really robust to umms and ahs.
Signal strength based VAD. They calc attention kvs while 
stt is running which is good. They say the biggest bottleneck 
is waiting for a sentence to TTS.
Their [roadmap](https://laion.ai/blog/bud-e/#a-roadmap-towards-empathic--natural-ai-voice-assistants) is very cool. 
They aim for a speech to speech model for chat.

### [Vocode](https://github.com/vocodedev/vocode-python)

Also have a hosted version [company](https://www.vocode.dev/).
Codebase is bad again. Not clear if they support interruptions.