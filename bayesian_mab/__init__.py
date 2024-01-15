"""Init file of LlamaIndex."""
from pathlib import Path

with open(Path(__file__).absolute().parents[0] / "VERSION") as _f:
    __version__ = _f.read().strip()


from bayesian_mab.multi_armed_bandit import BayesianMAB
from bayesian_mab.reward import BinaryReward, DiscreteReward
from bayesian_mab.finisher import BayesianFinisher
from bayesian_mab.arm import BayesianArm, BaseArm