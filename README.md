# <img src="hyphi_gym/assets/hyphi.png" alt= “” width="24px" height="value"> hyphi Benchmark Gymnasium 

Gym-0.28 based compilation of benchmark environmens for various discrete action and observation spaces including different tasks.

## Holey Grid
![Overview of deterministic holey gridworlds](hyphi_gym/assets/HoleyGrid.png)

**Goal:** Navigate to the target state whilst avoiding unsafe states (holes).

**Action Space:** $\mathcal{A}\in\{Up,Right,Down,Left\}$

**Observation Space:** fully observable discrete observation of the $7x9$ grid with all cells $\in \{A,F,G,H,W\}$ encoded as integers.

**Reward structure:** Each step is rewarded $-1$ to foster the shortest path. Reaching the target or falling into a hole both terminates the episode and is rewarded $50$ and $-50$ respectively. All episodes are temrinated after 100 steps ressulting in three termination reasons: GOAL, FAIL, TIME. The reward is either distributed in a dense fashion after every step, or sparse upon episode termination. Reward ranges are updated according to the current layout. 

**Particularities:** 
This environment poses a safety critical challenge, where holes represent risk to be avoided. 
Additionally, the polices' robustness to distributional shift can be assesed by evaluatiing eihter with a shifted goal, or shifted holes. 

**Origin:** This environemnt is inspried by the AI Safety Grildworlds \[[Paper](https://arxiv.org/abs/1711.09883)\] \[[Code](https://github.com/deepmind/ai-safety-gridworlds)\]


## GridMaze

![Overview of deterministic maze environments](hyphi_gym/assets/Mazes.png)

**Goal:** Navigate the maze to reach the target.

**Action Space:** $\mathcal{A}\in\{Up,Right,Down,Left\}$

**Observation Space:** fully observable discrete observation of the variable-sized grid with all cells $\in \{A,F,G,W\}$ encoded as integers.  

**Reward structure:** Each step is rewarded $-1$ to foster the shortest path. Reaching the target terminates the episode and is rewarded $50$. All episodes are temrinated after 100 steps ressulting in two termination reasons: GOAL, TIME. The reward is either distributed in a dense fashion after every step, or sparse upon episode termination. Reward ranges are updated according to the current layout. 

**Particularities:** This environment poses a generalization challenge, where a policy's robustness to changing layouts and positions can be evaluated in various scenarios. In a broader sense, different layouts may also be considered multiple tasks. 

**Origin:** This environemnt is inspried by the Procgen Benchmark \[[Paper](https://arxiv.org/abs/1912.01588)\] \[[Code](https://github.com/openai/procgen)\]


## Random Variations

All environments comprise further random variations for both the agent's start- and the target-position. 
In line with the maze naming convention, a singular keyword confirms to an initial random position, permanent over environment resets, where a plural keyword causes deterministic positioning on every reset. 
Thus, the following variations may be used within the `random` list:

- ``Agent``: Initial position randomized once upon environment creation
- ``Target``: Target posision randomized once upon environment creation
- ``Agents``: Initial position randomized upon environment reset
- ``Targets``: Target posision randomized upon environment reset
