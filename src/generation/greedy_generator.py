import torch
from src.generation.generator_output import GeneratorOutput
from src.generation.generator import Generator

class GreedyGenerator(Generator):
    
    def generate(self, inputs_ids: torch.Tensor) -> GeneratorOutput:
        
        messages = [{"role": "user", "content": inputs_ids}]
        input_text=self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs_ids = self.tokenizer(input_text, return_tensors = "pt")["input_ids"]
        len_initial_inputs_ids = inputs_ids.shape[1]
        
        while True:
            model_out = self.model.forward(inputs_ids)
            probs = torch.nn.functional.softmax(model_out.logits, dim=-1)
            next_token_id = torch.multinomial(probs[:, -1, :], num_samples=1)  # Sampling top token
            inputs_ids = torch.cat([inputs_ids, next_token_id], dim=-1)

            if next_token_id == 2 or inputs_ids.shape[1] - len_initial_inputs_ids >= self.max_new_token:
                return GeneratorOutput(
                    sequence = self.tokenizer.decode(inputs_ids.squeeze()[len_initial_inputs_ids:]), 
                    probabilites = probs.squeeze()[len_initial_inputs_ids-1:], 
                    generated_token_id = inputs_ids.squeeze()[len_initial_inputs_ids:]
                    )