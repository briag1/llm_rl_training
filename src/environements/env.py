import gymnasium as gym 
import numpy as np 
from typing import Optional

from src.environements.rewarder import Rewarder
from src.environements.parser import Parser
from src.environements.math_expression_generator import MathExpressionGenerator


class MathEnv(gym.Env):
    def __init__(self, rewarder: Rewarder, math_generator: MathExpressionGenerator) -> None:
        super().__init__()
        
        self.rewarder = rewarder 
        self.math_generator = math_generator
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.state = self.math_generator.generate_expression()
        return self.state
        
    def step(self, action: str):
        reward = self.rewarder.get_reward(action, self.state)
        answer = eval(self.state)
        return answer, reward, True, False, {}