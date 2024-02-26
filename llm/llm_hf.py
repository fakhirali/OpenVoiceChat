import torch
import numpy as np
import torch.nn.functional as F
import warnings
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from preprompts import sales_pre_prompt

warnings.filterwarnings("ignore")


class Chatbot:
    def __init__(self, model_name='stabilityai/stablelm-3b-4e1t', device='cuda'):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.device = device
        self.model.to(device)

    @torch.no_grad()	
    def generate_response_greedy(self, input_text, pre_prompt, break_word, 
                                max_length=100, temp=0.6, name='',
                                past_key_vals=None, next_id=None, verbose=True):

        if past_key_vals is None:
            inputs = self.tokenizer.encode(pre_prompt + input_text + '\n' + name, return_tensors="pt").to(self.device)
            response_ids = inputs
        else:
            inputs = self.tokenizer.encode(input_text + '\n' + name, return_tensors="pt").to(self.device)
            response_ids = torch.concat((next_id, inputs),dim=-1)
        output = ''
        last_n = ''
        if verbose:
            print(name, end='')
        response_text = ''
        for _ in (range(max_length)):
            out = self.model.forward(input_ids=response_ids, past_key_values=past_key_vals)
            next_token_id = torch.multinomial(F.softmax(out.logits[:, -1, :]/temp,  dim=-1), num_samples=1)
            past_key_vals = out.past_key_values
            response_ids = next_token_id
            # print([response_ids[0][-1].to('cpu')])
            output = self.tokenizer.decode([response_ids[0][-1].to('cpu')])
            if verbose:
                print(output, end='')
            response_text += output
            sys.stdout.flush()
            last_n += output
            last_n = last_n[-len(break_word):]
            if last_n == break_word or response_text.strip().endswith('<|endoftext|>') or response_text.strip().endswith('[END]'):
                break
        past_kv = past_key_vals
        next_id = response_ids
        return response_text, past_kv, next_id


if __name__ == "__main__":
    preprompt = sales_pre_prompt
    john = Chatbot()
    break_word = '[USER]'
    name = '[JOHN]'
    log = ''
    past_kv = None
    next_id = None
    print("type: exit, quit or stop to end the chat")
    print("Chat started:")
    while True:
        user_input = input(" ")
        if user_input.lower() in ["exit", "quit", "stop"]:
            break

        response,past_kv,next_id = john.generate_response_greedy(' ' + user_input, preprompt + log,
                                            break_word,max_length=100000, name=name,
                                            past_key_vals=past_kv, next_id=next_id, 
                                            verbose=True, temp=0.2)
        
        log += ' ' + user_input + '\n' + name + response
        print('------\n'+ log + '\n----------')