from src.environements.parser import Parser
from src.generation.generator import GeneratorOutput

class Rewarder:
    
    def __init__(self, parser: Parser)-> None:
        self.parser = parser
    def get_reward(self, generator_out: GeneratorOutput, reference_result: float) -> float:
        reward = 0
        try:
            parsed_output = self.parser.parse(generator_out)
            reward += 1
            result= float(parsed_output.result)
            reward -= abs(result-generator_out)/abs(result+generator_out)
        except:
            return reward - 1

            
            