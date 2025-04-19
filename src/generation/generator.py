import torch
from typing import Optional
from pydantic import Field
from pydantic.dataclasses import dataclass
from transformers import AutoTokenizer, LlamaForCausalLM
from src.generation.generator_output import GeneratorOutput

from abc import ABC, abstractmethod

@dataclass
class GeneratorConfig:
    model_name: str = Field(default= "HuggingFaceTB/SmolLM2-135M-Instruct")
    max_new_token: int = Field(default = 100)


    
class Generator(ABC):
    def __init__(self, generator_config : Optional[GeneratorConfig] = None):
        super().__init__()
        self.generator_config = GeneratorConfig() if generator_config is None else generator_config
        self.model = LlamaForCausalLM.from_pretrained(self.generator_config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.generator_config.model_name)
        self.max_new_token  = self.generator_config.max_new_token
    
    def to(self, device: str)->None:
        self.model = self.model.to(device)
        self.device = device
        
    @abstractmethod
    def generate(self, query: str) -> GeneratorOutput:
        ...
        
            
            
         
        
        