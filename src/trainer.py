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
    num_samples: int = Field(default = 200)
    eval_freq : int = Field(default= 50)
    eval_size : int = Field(default = 30)
    test_size : int = Field(default = 50)
    device: Device = Field(default= "cuda")
    lr: float = Field(default = 5e-5)

@dataclass 
class EvaluateInfo:
    losses: list[float] = Field(default = [])
    rewards: list[float] = Field(default = [])
    
@dataclass
class TrainingInfo:
    losses: list[float] = Field(default = [])
    rewards: list[float] = Field(default = [])
    eval_infos : list[EvaluateInfo] = Field(default = [])
    test_infos : Optional[EvaluateInfo] = None

    
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
        
    def evaluate(self, num_sample: int, index: int) -> EvaluateInfo:
        self.generator.eval()
        self.env.math_generator.set_split(index)
        eval_info = EvaluateInfo()
        for _ in tqdm(range(num_sample)):
            expression = self.env.reset()
            prompt = self.prompt_template.format(expression = expression)
            with torch.no_grad():
                out = self.generator.generate(prompt)
            log_probs = torch.log(out.probabilites[torch.arange(len(out.generated_token_id)),out.generated_token_id]).sum()
            _, reward, _, _, _ = self.env.step(out.sequence)
            loss= - reward*log_probs
            log_probs.detach()
            out.generated_token_id.detach()
            out.probabilites.detach()

            eval_info.losses.append(loss.detach())
            eval_info.rewards.append(reward)
        return eval_info
              
        
    def train(self,) -> TrainingInfo:
        
        training_info = TrainingInfo()
        optimizer = AdamW(self.generator.model.parameters(), lr=self.training_config.lr)
        
        for id_sample in tqdm(range(self.training_config.num_samples)):
            self.generator.train()
            if id_sample%self.training_config.eval_freq == 0:
                training_info.eval_infos.append(self.evaluate(self.training_config.eval_size, 1))
            expression = self.env.reset()
            prompt = self.prompt_template.format(expression = expression)
            out = self.generator.generate(prompt)
            log_probs = torch.log(out.probabilites[torch.arange(len(out.generated_token_id)),out.generated_token_id]).sum()
            _, reward, _, _, _ = self.env.step(out.sequence)
            loss= - reward*log_probs
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            log_probs.detach()
            loss = loss.detach()
            out.generated_token_id.detach()
            out.probabilites.detach()
            training_info.losses.append(loss)
            training_info.rewards.append(reward)
        training_info.test_infos = self.evaluate(self.training_config.eval_size, )
        return training_info
        
    
    