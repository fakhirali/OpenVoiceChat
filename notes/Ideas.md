# Some Ideas


## Speculative decoding of whisper output

We can stream the output of stt and have an LLM predict what the person will say next (also predicting what 
the bot will say). Do beam search and keep the beams that were correct according to the whisper output.
This is like "thinking while listening". 


## Interuptions

We constantly listen and transcribe and then stop tts. 
TTS has to stop on a word to make it natural. 
Also do we even stop. Sometimes the other person just says "yes" or "yeah". 
The stop recording policy is bad, it just stops recording after x number of silent seconds. 
There should be an EOS predictor for whisper, and we should stop using it, 
this will allow having pauses in the middle.
Use the whisper no speech tag as well.
Some good ideas [here](https://alphacephei.com/nsh/2023/09/22/time-brain-ctc-blank.html).

## Best Fast TTS

Bark is too slow, sounds alright. May hallucinate here and there. 
Xtts2 is slow. your_tts is fast but bad, glow_tts is better but not really. 
Jenny is alright, good quality audio.

## Thinking while speaking

We would want the agent to also think while the tts is running. 
The LLM response would be streamed out. The tts would get the LLM response and
stream the audio output. As the LLM buffer accumulates the audio buffer will also accumulate.

## How to get rid of the 2s silence detection delay

We need to understand when the human has stopped speaking, 
and wants the bot to speak. There could be many language cues.
To make a classifier utilizing the language cues we would need to
stream the audio and transcribe it in real time. I think one can make a 
classifier that gives the probability of the human stopping speaking.
VAD can also be used in conjunction to the probability to make a better 
decision. 





