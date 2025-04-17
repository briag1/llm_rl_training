import torch
from torch.optim import AdamW
from src.generation.generator import Generator
from src.environements.math_expression_generator import MathExpressionGenerator
from pydantic.dataclasses import dataclass
from pydantic import Field
from typing import Optional
from src.environements.env import MathEnv
from enum import StrEnum
from src.prompts import prompt_template
from tqdm import tqdm 
class Device(StrEnum):
    CUDA= "cuda"
    CPU ="cpu"
    
@dataclass
class TrainingConfig:
    num_samples: int = Field(default = 100)
    device: Device = Field(default= "cpu")
    lr: float = Field(default = 1e-4)

@dataclass
class TrainingInfo:
    losses: list[float] = Field(default = [])
    
    
class Trainer:
    def __init__(
        self, 
        generator: Generator,  
        env: MathEnv,
        prompt_template: str = prompt_template,
        training_config : Optional[TrainingConfig] = None
        ):
        self.training_config = training_config if  training_config is not None else TrainingConfig()
        self.generator = generator
        self.generator.to(self.training_config.device)
        self.env = env
        self.prompt_template = prompt_template
        
        
    def train(self,) -> TrainingInfo:
        training_info = TrainingInfo()
        optimizer = AdamW(self.generator.model.parameters(), lr=self.training_config.lr)
        for _ in tqdm(range(self.training_config.num_samples)):
            expression = self.env.reset()
            prompt = self.prompt_template.format(expression = expression)
            out = self.generator.generate(prompt)
            log_probs = torch.log(out.probabilites[torch.arange(len(out.generated_token_id)),out.generated_token_id]).sum()
            _, reward, _, _, _ = self.env.step(out.sequence)
            loss= - reward*log_probs
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss = loss.detach()
            training_info.losses.append(loss)
        return training_info
        
    
    