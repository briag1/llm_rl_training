
import random
import numpy as np
from abc import ABC, abstractmethod
from pydantic import Field
from typing import Optional
from pydantic.dataclasses import dataclass

@dataclass
class MathExpressionGeneratorConfig:
     low: int = Field(default = -100)
     high: int = Field( default = 100)
     splits: list[float] = Field(default=[0.8, 0.1, 0.1])
     seed: int = Field(default = 0)
    

class MathExpressionGenerator(ABC):
    def __init__(self, config: Optional[MathExpressionGeneratorConfig] = None):
        self.config = config if config is not None else MathExpressionGeneratorConfig()
        np.random.seed(self.config.seed)
        
    @abstractmethod
    def get_split_index(self, ) -> int:
        ...
    @abstractmethod    
    def set_split(self, index: int)-> None:
        ...
    
    def generate(self, num_expressions: int) -> list[str]:
        return [self.generate_expression() for _  in range(num_expressions)]
    
    @abstractmethod
    def generate_expression(self,) -> str:
        ...
        
    
class SimpleMathExpressionGenerator(MathExpressionGenerator):
    
    def __init__(self, config: Optional[MathExpressionGeneratorConfig] = None):
        super().__init__(config)
        self.config.splits
        normalized_split = np.array(self.config.splits)/np.sum(self.config.splits)
        digits = np.arange(start=self.config.low, stop = self.config.high)
        permutated_digits = np.random.permutation(digits)
        self.digits_splits = []
        current_index= 0
        for id_split, split in enumerate(normalized_split):
            if id_split != len(normalized_split) -1: 
                size = int(len(permutated_digits)*split)
                new_current_index = current_index+size
                self.digits_splits.append(permutated_digits[current_index: new_current_index])
                current_index = new_current_index
            else:
                self.digits_splits.append(permutated_digits[current_index: ])
        self.current_index = 0
        self.current_digits_split =  self.digits_splits[self.current_index]

    def get_split_index(self, ) -> int:
        return self.current_index
          
    def set_split(self, index: int)-> None:
        self.current_index = index
        self.current_digits_split = self.digits_splits[self.current_index]
    
    def generate_expression(self,) -> str:
        
        element = np.random.choice(self.current_digits_split, size = 2, replace = True).astype(str).tolist()
        symbols = ["+", "-", "/", "*"]
        id_symbol = np.random.choice(len(symbols))
        chosen_symbol = symbols[id_symbol]
        return chosen_symbol.join(element)
    
    
class ComplexMathExpressionGenerator(MathExpressionGenerator):
    def __init__(self, gamma : float = 0.1, clip_poisson: int = 5, max_depth: int = 30, start_symbol = "Expr"):
        self.grammar = {
            "Expr": [
                ["Expr", "+", "Expr"], 
                ["Expr", "-", "Expr"], 
                ["Expr"],
                ["(", "Expr", ")"], 
                ["Expr", "*", "Expr"], 
                ["-", "Expr"], 
                ["Expr", "/", "Expr"], 
                ["NUMBER"],
                ]
            }
        self.gamma = gamma
        self.clip_poisson = clip_poisson
        self.max_depth = max_depth
        self.start_symbol = start_symbol

    def generate_terminal(self, token):
        if token == "NUMBER":
            return str(random.randint(1, 100) if random.random() < 1-0.1/max(self.max_depth,5) else "0")
        return token

    def rand_space(self,):
        # Poisson distribution centered at λ=0.7, capped at 2
        return " " * min(np.random.poisson(self.gamma), self.clip_poisson)

    def join_with_random_space(self, parts):
        return parts[0] + "".join(self.rand_space() + p for p in parts[1:])
    
    def set_start_symbol(self, symbol: str)-> None:
        self.start_symbol = symbol
        
    def set_max_depth(self, max_depth: int) -> None:
        self.max_depth = max_depth
        
    def generate(self, num_expressions: int = 10)-> list[str]:
        [self.expand(symbol= self.start_symbol, depth=0) for _ in range(num_expressions)]
        
    def generate_expression(self,) -> str:
        return self.expand(self.start_symbol, depth= 0)
    
    def expand(self, symbol: str, depth:int = 0) -> str:
        if depth > self.max_depth:
            if symbol in ["Expr", "Term", "Factor"]:
                return self.generate("NUMBER", depth + 1)
        production = random.choice(self.grammar[symbol])
        parts = [self.generate(sym, depth + 1, self.max_depth) for sym in production]
        return self.join_with_random_space(parts)

    # Safe eval
    def safe_eval(self, expr):
        try:
            return eval(expr)
        except ZeroDivisionError:
            return "Division by zero"
        except Exception as e:
            return f"Error: {e}"


