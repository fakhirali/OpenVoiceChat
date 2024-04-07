from openai import OpenAI
from dotenv import load_dotenv
from llm.base import BaseChatbot
import os

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


if __name__ == "__main__":
    preprompt = 'You are a helpful assistant.'
    john = Chatbot_gpt(sys_prompt=preprompt)
    print("type: exit, quit or stop to end the chat")
    print("Chat started:")
    while True:
        user_input = input(" ")
        if user_input.lower() in ["exit", "quit", "stop"]:
            break

        response = john.generate_response(user_input)

        print(response)
