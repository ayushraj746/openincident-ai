# OpenIncident

**A Multi-Agent Reinforcement Learning Environment for Autonomous Incident Response**

---

## 1. Introduction

OpenIncident is a simulation environment designed to model real-world infrastructure failures and train intelligent agents to recover systems automatically.

In real production systems, outages are rarely simple. Engineers deal with incomplete data, conflicting signals, and rapidly changing conditions. OpenIncident tries to capture this complexity in a controlled environment so that AI agents can learn how to respond effectively.

The main idea is not just to “fix systems,” but to study how an agent **makes decisions under uncertainty**.

---

## 2. Problem Context

Modern AI systems can perform well on isolated tasks, but they struggle in situations where:

* information is incomplete or noisy
* multiple problems happen at the same time
* actions must be coordinated across different components
* early wrong decisions can make things worse

Incident response in production systems is a good example of this.

OpenIncident simulates such scenarios and allows an agent to:

* observe system state (partially)
* take actions
* receive feedback (reward)
* learn better strategies over time

---

## 3. System Overview

At a high level, the system works like this:

1. The environment starts in a degraded or failing state
2. The agent observes the system (with missing or noisy data)
3. The agent selects an action
4. The environment updates the system
5. A reward is given based on the outcome
6. This loop continues until the system recovers or fails

---

## 4. Environment Design

The core of the project is the `OpenIncidentEnv`.

### State Variables

The environment simulates key infrastructure signals:

* CPU usage (0–100)
* latency (response time in ms)
* memory usage (0–100)
* service health (healthy / degraded / down)
* network status (normal / slow / down)

---

### Key Features

#### 1. Partial Observability

Some values are randomly hidden (set to `None`).
This forces the agent to act even when it does not have full information.

#### 2. Stochastic Behavior

The system includes randomness:

* sudden spikes in CPU or latency
* gradual recovery over time

This prevents the environment from being predictable.

#### 3. Schema Drift

Sometimes the structure of the state changes. For example:

* `cpu_usage` may appear as `metrics.processor_load`
* `latency` may appear as `metrics.response_time`

This simulates real-world monitoring systems where data formats can change.

#### 4. Multi-step Recovery

Some problems require multiple steps to fix.
For example, a network issue may go from:

```
down → slow → normal
```

---

## 5. Action Space

The agent can perform actions such as:

* delegate infrastructure fixes (scale resources)
* clear memory or cache
* restart service
* restart network
* rollback deployment
* do nothing

Each action has a different effect on the system.

---

## 6. Multi-Agent Design

Instead of a single monolithic agent, the system uses a **multi-agent structure**.

### Incident Commander

This is the main decision-making component.
It:

* analyzes the current state
* identifies likely root causes
* chooses which agent should act
* selects the appropriate action

### SRE Agent

Handles infrastructure-related tasks such as:

* scaling resources
* managing memory
* handling latency issues

### Support Agent

Handles service-level and network-level recovery:

* restarting services
* fixing network issues

This separation reflects how real incident response teams operate.

---

## 7. Reward System

The reward engine is responsible for guiding learning.

### Positive Signals

The agent is rewarded when:

* CPU usage decreases
* latency decreases
* memory usage improves
* service becomes healthy
* network recovers

### Negative Signals

Penalties are given when:

* the same action is repeated unnecessarily
* actions do not improve the system
* wrong actions are taken for a given problem
* the agent does nothing without reason

### Terminal Reward

When the system is fully recovered, a bonus reward is given.

---

## 8. Reinforcement Learning Setup

The RL agent is trained using **Proximal Policy Optimization (PPO)** from Stable-Baselines3.

### Training Loop

The agent follows this cycle:

```
observe → act → receive reward → update policy
```

Over time, it learns which sequences of actions lead to faster recovery.

---

## 9. Gym Wrapper

A Gym-compatible wrapper (`GymOpenIncidentEnv`) is used to:

* convert the environment into numerical observations
* normalize values (0–1 range)
* encode categorical variables (network, service health)

This allows the environment to work with standard RL libraries.

---

## 10. Evaluation

We evaluate the system using three metrics:

* **Average Reward** → how effective the actions are
* **Average Steps** → how quickly the system recovers
* **Success Rate** → how often recovery is achieved

---

## 11. Rule-Based vs RL Comparison

To understand the effectiveness of learning, we compare:

### Rule-Based System

Uses fixed logic (if CPU high → scale, etc.)

### RL Agent

Learns behavior from experience

### Example Result

| Metric       | Rule-Based | RL Agent |
| ------------ | ---------- | -------- |
| Avg Reward   | 9.33       | 9.25     |
| Avg Steps    | 5.20       | 10.50    |
| Success Rate | 100%       | 90%      |

---

### Key Insight

The rule-based system performs better initially because it uses handcrafted logic.

The RL agent performs worse at first, which is expected. This shows that:

* the environment is challenging
* learning is non-trivial
* improvement requires more training

This is a strong indicator that the environment is realistic.

---

## 12. Project Structure

```
openincident/
│
├── env/                  # environment logic
├── agents/               # commander, SRE, support agents
├── reward/               # reward calculation
├── training/             # training, evaluation, comparison
├── inference_engine/     # rule-based execution
├── analysis/             # optional plotting
│
├── OPENINCIDENT/
│   └── openenv.yaml      # environment metadata
│
├── requirements.txt
├── README.md
```

---

## 13. How to Run

### Install dependencies

```
pip install -r requirements.txt
```

### Train RL agent

```
python -m training.train_rl
```

### Evaluate RL agent

```
python -m training.evaluate_rl
```

### Compare RL vs Rule-Based

```
python -m training.compare
```

---

## 14. Limitations

* RL agent needs more training to outperform rule-based system
* environment is simulated (not connected to real infrastructure)
* explainability of decisions is limited

---

## 15. Future Improvements

* multi-agent RL (each agent learns independently)
* integration with real logs and monitoring data
* better explainability (“why this action was chosen”)
* distributed incident simulation

---

## 16. Conclusion

OpenIncident is designed as a realistic environment for studying how AI agents handle complex, uncertain situations.

It shows that:

* rule-based systems are strong but limited
* learning-based systems require time but can adapt
* realistic environments are necessary for meaningful progress

The focus of this project is not just performance, but creating a **credible training environment for intelligent decision-making systems**.

---
