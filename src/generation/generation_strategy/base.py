from abc import ABC, abstractmethod
from pydantic.dataclasses import dataclass

@dataclass
class GenerationInput:
    query: str
    prompt: str
    
class GenerationStrategy(ABC):
    
    @abstractmethod
    def generate(self, inputs_ids: str, model, **kwargs) -> str:
        ...