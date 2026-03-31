from enum import Enum
from typing import List, Union


class CustomEnum(Enum):
    @classmethod
    def get_modes(cls, modes_str: Union[str, List[str]]):
        if isinstance(modes_str, str):
            return cls(modes_str)
        return [cls(mode_str) for mode_str in modes_str]


class TemporalPoolingType(CustomEnum):
    LEARNABLE = "learnable"
    AVERAGE = "average"


class EmbeddingAddType(CustomEnum):
    CONCAT = "concat"
    SUM = "sum"
