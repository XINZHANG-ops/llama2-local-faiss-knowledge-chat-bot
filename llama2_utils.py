import torch
from transformers import (
    BitsAndBytesConfig
)
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer)
from typing import Iterator
from threading import Thread
# from huggingface_hub import login
# login(token="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")

class llama2:
    def __init__(self, model_id):
        # pip install git+https://github.com/huggingface/transformers.git
        # https://github.com/jllllll/bitsandbytes-windows-webui
        # python -m pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui

        ################################################################################
        # bitsandbytes parameters
        ################################################################################
        # Activate 4-bit precision base model loading
        use_4bit = True
        # Compute dtype for 4-bit base models
        bnb_4bit_compute_dtype = "float16"
        # Quantization type (fp4 or nf4)
        bnb_4bit_quant_type = "nf4"
        # Activate nested quantization for 4-bit base models (double quantization)
        use_nested_quant = True
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        if compute_dtype == torch.float16 and use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16: accelerate training with bf16=True")
                print("=" * 80)
        # Load tokenizer and model with QLoRA configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,  # compute_dtype, torch.bfloat16
            bnb_4bit_use_double_quant=use_nested_quant,
        )

        # self.DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
        # If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
        self.DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant, try your best to answer whatever people ask."
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True
        )
        self.MAX_NEW_TOKENS = 2048
        self.DEFAULT_MAX_NEW_TOKENS = 1024
        self.MAX_INPUT_TOKEN_LENGTH = 4000
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
        self.history = [("system", self.DEFAULT_SYSTEM_PROMPT)]

    def get_prompt_adjust_input_length(self, message: str) -> str:
        chat_history = self.history.copy()
        # 计算系统信息的长度
        system_prompt_length = len(
            self.tokenizer.encode(f'<s>[INST] <<SYS>>\n{self.DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n'))

        # 计算当前历史消息的总长度，包括系统信息和最新问题
        total_history_length = system_prompt_length + len(self.tokenizer.encode(message + " [/INST]")) + \
                               sum([len(self.tokenizer.encode(user_input + " [/INST] ")) + len(
                                   self.tokenizer.encode(response + " </s><s>[INST] "))
                                    for user_input, response in chat_history])

        # 如果历史消息长度超过上限，丢弃旧的历史消息，只保留最新的历史
        while total_history_length > self.MAX_INPUT_TOKEN_LENGTH:
            user_input, response = chat_history.pop(0)
            total_history_length -= len(self.tokenizer.encode(user_input + " [/INST] ")) + len(
                self.tokenizer.encode(response + " </s><s>[INST] "))

        texts = [f'<s>[INST] <<SYS>>\n{self.DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n']
        # The first user input is _not_ stripped
        do_strip = False
        for user_input, response in chat_history:
            user_input = user_input.strip() if do_strip else user_input
            do_strip = True
            texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
        message = message.strip() if do_strip else message
        texts.append(f'{message} [/INST]')
        return ''.join(texts)

    # def get_prompt(self, message: str, chat_history: list[tuple[str, str]],
    #                system_prompt: str) -> str:
    #     texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    #     # The first user input is _not_ stripped
    #     do_strip = False
    #     for user_input, response in chat_history:
    #         user_input = user_input.strip() if do_strip else user_input
    #         do_strip = True
    #         texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    #     message = message.strip() if do_strip else message
    #     texts.append(f'{message} [/INST]')
    #     return ''.join(texts)
    #
    # def get_input_token_length(self, message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> int:
    #     prompt = self.get_prompt(message, chat_history, system_prompt)
    #     input_ids = self.tokenizer([prompt], return_tensors='np', add_special_tokens=False)['input_ids']
    #     return input_ids.shape[-1]
    #
    # def check_input_token_length(self, message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> None:
    #     input_token_length = self.get_input_token_length(message, chat_history, system_prompt)
    #     if input_token_length > self.MAX_INPUT_TOKEN_LENGTH:
    #         print(
    #             f'The accumulated input is too long ({input_token_length} > {self.MAX_INPUT_TOKEN_LENGTH}). Clear your chat history and try again.')

    def reset_history(self):
        self.history = [("system", self.DEFAULT_SYSTEM_PROMPT)]

    def update_history(self, role: str, message: str):
        self.history.append((role, message))

    def run(self, message: str,
            temperature: float = 0.8,
            top_p: float = 0.95,
            top_k: int = 50) -> Iterator[str]:
        prompt = self.get_prompt_adjust_input_length(message)
        inputs = self.tokenizer([prompt], return_tensors='pt', add_special_tokens=False).to('cuda')

        streamer = TextIteratorStreamer(self.tokenizer,
                                        timeout=10.,
                                        skip_prompt=True,
                                        skip_special_tokens=True)
        generate_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=self.MAX_NEW_TOKENS,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=1,
        )
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        outputs = []
        for text in streamer:
            outputs.append(text)
            yield ''.join(outputs).lstrip()


# def demo():
#     llama2_api = llama2('meta-llama/Llama-2-7b-chat-hf')
#     prompt = "what is 1 + 1"
#     generator = llama2_api.run(prompt)
#     previous_texts = ""
#     for response in generator:
#         print(response[len(previous_texts):], end='')
#         previous_texts = response
#
#     llama2_api.update_history('user', prompt)
#     llama2_api.update_history('assistant', previous_texts.strip())
#     print()
#     prompt2 = "what if plus another 1?"
#     generator = llama2_api.run(prompt2)
#     previous_texts = ""
#     for response in generator:
#         print(response[len(previous_texts):], end='')
#         previous_texts = response
