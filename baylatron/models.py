from typing import Dict, List, Union

from pydantic import BaseModel


class BayesianFinishIteration(BaseModel):
    arm_comp: str
    arm_base: str
    prob: float
    lift: float
    iteration: int


class Reward(BaseModel):
    current_reward: Union[float, int]
