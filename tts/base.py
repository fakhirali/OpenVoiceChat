import sounddevice as sd
import torch
import re
from time import monotonic
import queue
import threading


def remove_words_in_brackets_and_spaces(text):
    # Pattern to match optional spaces, content inside square brackets, and optional spaces again
    # This will handle spaces before and after the brackets
    pattern = r'\s*\[.*?\]\s*'
    # Remove the matching content and replace it with a single space to avoid multiple spaces
    cleaned_text = re.sub(pattern, ' ', text)
    # Optionally, you might want to trim leading or trailing spaces
    cleaned_text = cleaned_text.strip()
    return cleaned_text


class BaseMouth:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.sentence_stop_pattern = r'[.?](?=\s+\S)'
        self.interrupted = False

    @torch.no_grad()
    def run_tts(self, text):
        raise NotImplementedError('This method should be implemented by the subclass')

    def say_text(self, text):
        output = self.run_tts(text)
        sd.play(output, samplerate=self.sample_rate)
        sd.wait()

    def say(self, audio_queue, listen_interruption_func):
        self.interrupted = False
        while True:
            output = audio_queue.get()
            if output is None:
                break
            # get the duration of audio
            duration = len(output) / self.sample_rate
            sd.play(output, samplerate=self.sample_rate)
            interruption = listen_interruption_func(duration)
            if interruption:
                sd.stop()
                self.interrupted = True
                break
            else:
                sd.wait()

    def say_multiple(self, text, listen_interruption_func):
        pattern = r'[.?!]'
        sentences = re.split(pattern, text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        print(sentences)
        audio_queue = queue.Queue()
        say_thread = threading.Thread(target=self.say, args=(audio_queue, listen_interruption_func))
        say_thread.start()
        for sentence in sentences:
            output = self.run_tts(sentence)
            audio_queue.put(output)
            if self.interrupted:
                break
        audio_queue.put(None)
        say_thread.join()

    def say_multiple_stream(self, text_queue, listen_interruption_func, interrupt_queue):
        response = ''
        all_response = ''
        audio_queue = queue.Queue()
        say_thread = threading.Thread(target=self.say, args=(audio_queue, listen_interruption_func))
        say_thread.start()
        while True:
            text = text_queue.get()
            if text is None:
                response = remove_words_in_brackets_and_spaces(response).strip()
                if response.strip() != '':
                    output = self.run_tts(response)
                    audio_queue.put(output)
                if self.interrupted:
                    interrupt_queue.put(True)
                text_queue.put(all_response)
                break
            response += text
            all_response += text
            if bool(re.search(self.sentence_stop_pattern, response)):
                sentences = re.split(self.sentence_stop_pattern, response, maxsplit=1)
                sentence = sentences[0]
                response = sentences[1]
                sentence = remove_words_in_brackets_and_spaces(sentence).strip()
                output = self.run_tts(sentence)
                audio_queue.put(output)
                if self.interrupted:
                    text_queue.put(all_response)
                    interrupt_queue.put(True)
                    break
        audio_queue.put(None)
        say_thread.join()

