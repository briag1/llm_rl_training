import torch 
from dataclasses import dataclass


@dataclass
class GeneratorOutput:
    sequence: str
    probabilites: torch.Tensor
    generated_token_id : torch.Tensor