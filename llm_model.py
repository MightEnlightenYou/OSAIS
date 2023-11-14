from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class LLMModel:
    def __init__(self, model_path, temperature, top_k, top_p, repetition_penalty, length_penalty):
        self.model_path = model_path
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty

        self.llm = LlamaCpp(
            model_path=self.model_path,
            n_gpu_layers=1,
            n_batch=512,
            n_ctx=4096,
            max_tokens=1000,
            f16_kv=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            threads=4,
            max_length=200,
        )

    def generate_response(self, prompt):
        return self.llm(prompt)