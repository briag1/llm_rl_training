from src.environements.parser import Parser
from src.generation.generator import GeneratorOutput
from pydantic.dataclasses  import dataclass
from pydantic import Field
from typing import Optional

@dataclass
class RewarderConfig:
    eps : float = Field(default = 1e-3)
    
    
    
class Rewarder:
    def __init__(self, parser: Parser, config: Optional[RewarderConfig] = None,  )-> None:
        self.config = config
        self.parser = parser
        
    def get_reward(self, generator_out: GeneratorOutput, reference_result: float) -> float:
        reward = 0
        try:
            parsed_output = self.parser.parse(generator_out)
            reward += 2
            result = float(parsed_output.result)
            percentage_distance = (result-reference_result)**2/(abs(reference_result)+self.config.eps)**2
            reward += self.config.eps/(percentage_distance + self.config.eps)
        except:
            return reward - 1

            
            