import re
from typing import Optional
from pydantic.dataclasses import dataclass
from pydantic import Field

class ParsingError(Exception):
    """Custom exception for parsing errors."""
    pass
@dataclass
class ParsingConfig:
    reasoning_heading: str = Field(default= "Reasonning:")
    result_heading:str = Field(default= "Result:")
    assistant_start: str = Field(default = "<|im_start|>assistant")
    assistant_end: str = Field(default = "<|im_end|>")
    
class Parser:
    def __init__(self, parsing_config: Optional[ParsingConfig] = None):
        self.parsing_config = ParsingConfig() if parsing_config is None else parsing_config
         
    def parse(self, input_str: str):
        input_str = input_str.strip()
        if input_str.startswith(self.parsing_config.assistant_start):
            input_str.replace(self.parsing_config.assistant_start, "")
        else:
            raise ParsingError
        if input_str.endswith(self.parsing_config.assistant_end):
            input_str.replace(self.parsing_config.assistant_end, "")
        else:
            raise ParsingError
        splitted_output = input_str.split(self.parsing_config.result_heading)
        if len(splitted_output) != 2:
            raise ParsingError
        splitted_resonning = splitted_output[0].split(self.parsing_config.reasoning_heading)
        if len(splitted_resonning) != 2:
            raise ParsingError
        return ParsedOutput(splitted_resonning[-1].strip(), splitted_output[-1].strip())
 
@dataclass
class ParsedOutput:
    reasoning : str
    result: str