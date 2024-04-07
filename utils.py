import numpy as np
import pyaudio
import audioop

CHUNK = int(1024*2)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


def make_stream():
    p = pyaudio.PyAudio()
    return p.open(format=FORMAT,
                  channels=CHANNELS,
                  rate=RATE,
                  input=True,
                  frames_per_buffer=CHUNK)


# def record_audio(record_seconds=100):
#     # yield audio frames
#
#     RECORD_SECONDS = record_seconds
#
#     stream = make_stream()
#
#     for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#         data = stream.read(CHUNK)
#         yield data
#


def record_interruption_parallel(vad, listen_queue):
    #listen for interruption untill the queue is not empty
    frames = []
    stream = make_stream()
    while True:
        a = listen_queue.get()
        if a is None:
            break
        data = stream.read(CHUNK)
        frames.append(data)
        contains_speech = vad.contains_speech(frames[int(RATE / CHUNK) * -2:])
        if contains_speech:
            stream.close()
            frames = np.frombuffer(b''.join(frames), dtype=np.int16)
            frames = frames / (1 << 15)
            return frames.astype(np.float32)
    stream.close()
    return None


def record_interruption(vad, recond_seconds=100):
    print("* recording for interruption")
    frames = []
    stream = make_stream()
    for _ in range(0, int(RATE / CHUNK * recond_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)
        contains_speech = vad.contains_speech(frames[int(RATE / CHUNK) * -2:])
        if contains_speech:
            stream.close()
            frames = np.frombuffer(b''.join(frames), dtype=np.int16)
            frames = frames / (1 << 15)
            return frames.astype(np.float32)
    stream.close()
    return None


def record_user(silence_seconds, vad):
    frames = []

    started = False
    one_second_iters = int(RATE / CHUNK)
    stream = make_stream()
    print("* recording")

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        contains_speech = vad.contains_speech(frames[int(-one_second_iters * silence_seconds):])
        if not started and contains_speech:
            started = True
            print("*listening to speech*")
        if started and contains_speech is False:
            break
    stream.close()

    print("* done recording")

    # creating a np array from buffer
    frames = np.frombuffer(b''.join(frames), dtype=np.int16)

    # normalization see https://discuss.pytorch.org/t/torchaudio-load-normalization-question/71470
    frames = frames / (1 << 15)

    return frames.astype(np.float32)
