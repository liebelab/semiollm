# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC
import torch
from openai import OpenAI
import transformers
from fireworks.client import Fireworks
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers import pipeline


FIREWORKS_KEY =  'XXXXXXX'  #replace with your key
openai_key = 'XXXXXXX' #replace with your key

class LMMBaseModel(ABC):
    """
    Abstract base class for language model interfaces.

    This class provides a common interface for various language models and includes methods for prediction.

    Parameters:
    -----------
    model : str
        The name of the language model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').

    Methods:
    --------
    predict(input_text, **kwargs)
        Generates a prediction based on the input text.
    __call__(input_text, **kwargs)
        Shortcut for predict method.
    """
    def __init__(self, model_name, max_new_tokens, temperature, device='auto'):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device

    def predict(self, input_text, **kwargs):
        if self.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.device
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)

        outputs = self.model.generate(input_ids, 
                                     max_new_tokens=self.max_new_tokens, 
                                     temperature=self.temperature,
                                     do_sample=True,
                                     **kwargs)
        
        out = self.tokenizer.decode(outputs[0])
        return out

    def __call__(self, input_text, **kwargs):
        return self.predict(input_text, **kwargs)


class MixtralModel(LMMBaseModel):
    """
    Language model class for the Mixtral model.

    Inherits from LMMBaseModel and sets up the Mixtral language model for use.

    Parameters:
    -----------
    model : str
        The name of the Mixtral model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').
    dtype: str
        The dtype to use for inference (default is 'auto').
    """
    def __init__(self, model_name, max_new_tokens, temperature, top_p, top_k):
        super(MixtralModel, self).__init__(model_name, max_new_tokens, temperature)
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def predict(self, input_text, **kwargs):
        client = Fireworks(api_key=FIREWORKS_KEY)
        message = input_text
        #read kwargs
        temperature = kwargs['temperature'] if 'temperature' in kwargs else self.temperature
        max_tokens = kwargs['max_new_tokens'] if 'max_new_tokens' in kwargs else self.max_new_tokens
        response = client.chat.completions.create(                    
                    model="accounts/fireworks/models/mixtral-8x7b-instruct",
                    messages=message,
                    stop=["<|im_start|>","<|im_end|>","<|endoftext|>"],
                    n=1,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    presence_penalty=0,
                    frequency_penalty=0,
                    prompt_truncate_len=1024,
                    context_length_exceeded_behavior="truncate",
                    temperature=temperature,
                    max_tokens=max_tokens
                )
        result = response.choices[0].message.content
        return result

class OpenAIModel(LMMBaseModel):
    """
    Language model class for interfacing with OpenAI's GPT models.

    Inherits from LMMBaseModel and sets up a model interface for OpenAI GPT models.

    Parameters:
    -----------
    model : str
        The name of the OpenAI model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    system_prompt : str
        The system prompt to be used (default is None).
    openai_key : str
        The OpenAI API key (default is None).

    Methods:
    --------
    predict(input_text)
        Predicts the output based on the given input text using the OpenAI model.
    """
    def __init__(self, model_name, max_new_tokens, temperature, top_p, system_prompt):
        super(OpenAIModel, self).__init__(model_name, max_new_tokens, temperature)
        self.openai_key = openai_key
        self.system_prompt = system_prompt
        self.top_p = top_p

    def predict(self, input_text, **kwargs):
        
        client = OpenAI(api_key=openai_key)
        
        system_messages = self.system_prompt
        # if self.system_prompt is None:
        #     system_messages = self.system_prompt
        # else:
        #     system_messages = {'role': "system", 'content': self.system_prompt}
        
        if isinstance(input_text, list):
            messages = input_text
        elif isinstance(input_text, dict):
            messages = [input_text]
        else:
            messages = [{"role": "user", "content": input_text}]
        
        messages.insert(0, system_messages)
    
        # extra parameterss
        n = kwargs['n'] if 'n' in kwargs else 1
        temperature = kwargs['temperature'] if 'temperature' in kwargs else self.temperature
        top_p = kwargs['top_p'] if 'top_p' in kwargs else self.top_p
        max_new_tokens = kwargs['max_new_tokens'] if 'max_new_tokens' in kwargs else self.max_new_tokens
        
        if 'davinci' in self.model_name or 'instruct' in self.model_name:
            messages = messages[0]['content'] + messages[1]['content']
            response = client.completions.create(
                model=self.model_name,
                prompt=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                n=n,
            )
            result = response.choices[0].text
        else:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                n=n,
            )        
            if n > 1:
                result = [choice.message.content for choice in response.choices]
            else:
                result = response.choices[0].message.content
            
        return result
                