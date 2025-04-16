import torch

from transformers import PreTrainedTokenizer, LlamaForCausalLM
from src.generation.generation_strategy.base import GenerationStrategy
from src.generation.generator import GeneratorOutput


class GreedyGeneration(GenerationStrategy):
    
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        max_new_token: int
        ) -> None:
        self.tokenizer = tokenizer
        self.max_new_token = max_new_token
    
    def generate(self, inputs_ids: torch.Tensor, model: LlamaForCausalLM, max_new_token: int) -> GeneratorOutput:
        len_initial_inputs_ids = inputs_ids.shape[1]
        while True:
            model_out = model.forward(inputs_ids)
            probs = torch.nn.functional.softmax(model_out.logits, dim=-1)
            next_token_id = torch.multinomial(probs[:, -1, :], num_samples=1)  # Sampling top token
            inputs_ids = torch.cat([inputs_ids, next_token_id], dim=-1)

            if next_token_id == 2 or inputs_ids.shape[1] - len_initial_inputs_ids >= max_new_token:
                return GeneratorOutput(
                    sequence = self.tokenizer.decode(inputs_ids.squeeze()[len_initial_inputs_ids:]), 
                    probabilites = probs.squeeze()[len_initial_inputs_ids-1:], 
                    generated_token_id = inputs_ids.squeeze()[len_initial_inputs_ids:]
                    )