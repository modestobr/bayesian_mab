# Bayesian Multi Armed Bandit


## 🚀 Overview

A Bayesian Multi-Armed Bandit is a statistical model used in decision-making processes under uncertainty. It is a variation of the classic multi-armed bandit problem, where you have multiple options (each represented as an arm of a bandit, or slot machine) and you must choose which to pursue to maximize your rewards. The Bayesian aspect of this model comes into play by using Bayesian inference to update the probability distribution of the rewards of each arm based on prior knowledge and observed outcomes. This approach allows for a more nuanced and dynamically adaptive decision-making process, as the model continuously updates its beliefs and predictions about the performance of each option in real time. It's especially useful in scenarios where the environment changes or when dealing with limited information.

Between several use cases, we can highlight
- Online adversising
- Website optimization
- Personalization
- Clinical trials

## 💻 Example Usage

```python
from multi_armed_bandit import BayesianMAB, BinaryReward

binary_reward = BinaryReward()

bayesian_mab = BayesianMAB(
    arms=[
        BayesianArm(index=0, arm_name="Ad #1"),
        BayesianArm(index=1, arm_name="Ad #2"),
        BayesianArm(index=2, arm_name="Ad #3"),
    ]
)

for i in range(4):
    binary_reward.update_reward(np.random.binomial(1, p=0.9))
    bayesian_mab.update_arm(chosen_arm=1, reward_agent=binary_reward)

for i in range(1500):
    binary_reward.update_reward(np.random.binomial(1, p=0.3))
    bayesian_mab.update_arm(chosen_arm=1, reward_agent=binary_reward)

for i in range(1500):
    binary_reward.update_reward(np.random.binomial(1, p=0.9))
    bayesian_mab.update_arm(chosen_arm=2, reward_agent=binary_reward)

flg_end, winner_arm = bayesian_mab.check_for_end(winner_prob_threshold=0.80)

print("Is there a winner? {}. Winner: {}".format(flg_end, winner_arm))
```

## Acknowledgments and References


- Cook, J., 2005. **Exact calculation of beta inequalities**. Houston: University of Texas, MD Anderson Cancer Center. Available [here](https://www.johndcook.com/UTMDABTR-005-05.pdf)
- Slivkins, A., 2019. **Introduction to multi-armed bandits**. Foundations and Trends® in Machine Learning, 12(1-2), pp.1-286. Available [here](https://www.nowpublishers.com/article/Details/MAL-068)
- White, J., 2013. **Bandit algorithms for website optimization.** " O'Reilly Media, Inc.".
- Praise on Vincenzo Lavorini for [this](https://towardsdatascience.com/bayesian-a-b-testing-with-python-the-easy-guide-d638f89e0b8a) Towards Data Science blog post.

