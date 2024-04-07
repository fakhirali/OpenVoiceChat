from preprompts import call_pre_prompt
from llama_cpp import Llama
from llm.base import BaseChatbot


class Chatbot_llama(BaseChatbot):
    def __init__(self, model_path='models/llama-2-7b.Q4_K_M.gguf', device='cuda', sys_prompt='',
                 break_words=None, name='[JOHN]'):
        if break_words is None:
            break_words = ['[USER]', '[END]']
        self.model = Llama(model_path=model_path, n_ctx=4096,
                           n_gpu_layers=-1 if device == 'cuda' else 0,
                           verbose=False)
        self.sys_prompt = sys_prompt
        self.break_words = break_words
        self.name = name
        self.messages = []
        self.messages.append(self.sys_prompt)

    def run(self, input_text):
        self.messages.append(input_text + '\n' + self.name)
        out = self.model.create_completion(''.join(self.messages),
                                           max_tokens=1000, stream=True)
        response_text = ''
        for o in out:
            text = o['choices'][0]['text']
            response_text += text
            yield text
            if any([response_text.strip().endswith(break_word) for break_word in self.break_words]):
                break

    def post_process(self, response):
        self.messages.append(response)
        return response


if __name__ == "__main__":
    preprompt = call_pre_prompt
    john = Chatbot_llama(model_path='../models/llama-2-7b.Q4_K_M.gguf', sys_prompt=preprompt)
    print("type: exit, quit or stop to end the chat")
    print("Chat started:")
    while True:
        user_input = input(" ")
        if user_input.lower() in ["exit", "quit", "stop"]:
            break

        response = john.generate_response(user_input)

        print(response)
