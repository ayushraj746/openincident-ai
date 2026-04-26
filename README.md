# OpenIncident — Multi-Agent Incident Response Environment

OpenIncident is a simulation environment designed to model real-world infrastructure failures and train intelligent agents to autonomously recover systems using reinforcement learning.

The project focuses on decision-making under uncertainty, where agents must act with incomplete information, handle multiple failures, and optimize recovery over time.

---

## 1. Problem

Modern distributed systems are complex and failure-prone. Issues such as CPU spikes, memory leaks, latency increases, and network disruptions often occur simultaneously and evolve over time.

Traditional incident response is:

* manual
* reactive
* dependent on predefined rules

This approach does not scale well and struggles in unpredictable environments.

OpenIncident explores whether AI agents can:

* observe partially available system signals
* take corrective actions
* learn optimal recovery strategies through interaction

---

## 2. Environment Overview

The environment simulates a production system that evolves step by step.

At each timestep:

1. The system starts in a degraded or failing state
2. The agent observes the system (with partial information)
3. The agent selects an action
4. The environment updates the system
5. A reward is assigned
6. The loop continues until recovery or failure

This setup creates a sequential decision-making problem suitable for reinforcement learning.

---

## 3. Environment Design

### State Variables

The environment includes key infrastructure signals:

* CPU usage (0–100)
* Memory usage (0–100)
* Latency (ms)
* Service health (healthy / degraded / down)
* Network status (normal / slow / down)

The observation is converted into a fixed 7-dimensional vector for RL compatibility.

---

### Key Characteristics

#### Partial Observability

Some values may be hidden or missing, forcing the agent to act without complete information.

#### Stochastic Behavior

The system evolves with randomness:

* sudden spikes in CPU or latency
* gradual recovery

#### Schema Variations

State representation may change dynamically, simulating real-world monitoring inconsistencies.

#### Multi-step Recovery

Certain failures require sequences of actions rather than a single fix.

---

## 4. Action Space

The agent can perform actions such as:

* scale infrastructure
* clear memory/cache
* restart services
* restart network
* rollback deployment
* no-op (do nothing)

Each action affects the system differently depending on the state.

---

## 5. Multi-Agent Architecture

The system follows a structured multi-agent design:

### Commander Agent

* Central decision-maker
* Chooses actions or delegates tasks
* Supports both rule-based and RL modes

### SRE Agent

* Handles infrastructure and scaling
* Optimizes CPU, memory, and latency

### Support Agent

* Manages service and network recovery

### Security Agent

* Handles anomaly detection and mitigation

This structure reflects real-world incident response workflows.

---

## 6. Reward System

The reward function guides agent learning.

### Positive Rewards

* reduction in CPU, latency, and memory usage
* recovery of services
* network stabilization

### Negative Rewards

* ineffective or incorrect actions
* repeated unnecessary actions
* system degradation
* inaction when intervention is required

### Terminal Reward

A bonus reward is given when the system fully recovers.

The reward is step-based, providing continuous feedback during training.

---

## 7. Reinforcement Learning Setup

* Algorithm: Proximal Policy Optimization (PPO)
* Framework: Stable-Baselines3
* Observation space: 7 features
* Action space: discrete

The agent learns through interaction:

observe → act → receive reward → update policy

---

## 8. Gym-Compatible Wrapper

A Gym-compatible wrapper converts the environment into a format usable by RL libraries.

It:

* normalizes numerical values
* encodes categorical variables
* ensures consistent observation shape

This enables seamless training and evaluation.

---

## 9. Evaluation

We evaluate performance using:

* Average Reward
* Average Steps to Recovery
* Success Rate

---

## 10. Rule-Based vs RL Comparison

Two approaches are compared:

### Rule-Based System

Uses predefined logic (e.g., high CPU → scale resources)

### RL Agent

Learns behavior through experience

### Observations

* Rule-based system performs well initially
* RL agent starts weaker but improves with training
* RL demonstrates more adaptive behavior under uncertainty

---

## 11. Training Results

The following plot shows reward trends during simulation:

![Reward Graph](assets/reward_graph.png)

Key observations:

* RL rewards become more stable over time
* Fewer extreme failures compared to rule-based behavior
* Smoother recovery patterns

This indicates that the agent is learning meaningful policies.

---

## 12. Demo

Live interactive demo:

Hugging Face Space:
(Add link after deployment)

The demo includes:

* rule-based simulation
* RL-based simulation
* side-by-side comparison
* real-time metrics and reward visualization

---

## 13. Project Structure

```
openincident/
│
├── app.py
├── requirements.txt
│
├── env/
├── agents/
├── reward/
├── training/
├── inference_engine/
│
├── ppo_incident_model/
```

---

## 14. How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
streamlit run app.py
```

---

## 15. Training

Training script:

```
training/train_rl.py
```

Trained model:

```
ppo_incident_model/
```

---

## 16. Key Design Decisions

* fixed observation mismatch (12 → 7 features)
* step-based reward instead of cumulative reward
* reward smoothing for stable visualization
* modular multi-agent architecture

---

## 17. Limitations

* RL agent requires more training to outperform rule-based logic
* environment is simulated, not connected to real systems
* limited explainability of agent decisions

---

## 18. Future Work

* multi-agent reinforcement learning
* more complex incident scenarios
* integration with real monitoring data
* improved interpretability of decisions

---

## 19. Conclusion

OpenIncident provides a realistic environment for studying how intelligent agents handle complex system failures.

It demonstrates that:

* rule-based systems are reliable but rigid
* learning-based systems are adaptive but require training
* realistic environments are essential for meaningful progress

The focus is on building a credible training environment for autonomous decision-making systems.

---

## 20. Additional Resources

* Training script: `training/train_rl.py`
* Model: `ppo_incident_model/`
* Demo: (Hugging Face link)
* Video/Blog: (Add link here)

---

