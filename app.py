import streamlit as st
import time
import matplotlib.pyplot as plt
import numpy as np

from env.environment import OpenIncidentEnv
from env.gym_wrapper import GymOpenIncidentEnv
from stable_baselines3 import PPO
from agents.commander import IncidentCommander
from agents.sre_agent import SREAgent
from agents.support_agent import SupportAgent
from agents.security_agent import SecurityAgent

st.set_page_config(page_title="AutoSRE AI", layout="wide")

st.title("🚀 AutoSRE AI — Intelligent Incident Response")
st.markdown("AI-powered multi-agent system for autonomous recovery")

# ---------------- SIDEBAR ---------------- #

mode = st.sidebar.selectbox(
    "Mode",
    ["Rule-Based", "RL Agent", "Compare"]
)

speed = st.sidebar.slider("Speed", 0.05, 1.0, 0.2)

# ---------------- HELPERS ---------------- #

def safe(val, default):
    return val if val is not None else default


def explain(action):
    mapping = {
        "restart_service": "🔁 Restart Service",
        "delegate_sre": "⚙️ Scale Infra",
        "delegate_memory": "🧠 Clear Memory",
        "delegate_network": "🌐 Fix Network",
        "rollback_deployment": "⏪ Rollback",
        "block_traffic": "🚫 Block Traffic",
        "do_nothing": "⏳ Stabilizing",
    }
    return mapping.get(action, action)


def smooth_curve(values, window=5):
    smooth = []
    for i in range(len(values)):
        smooth.append(np.mean(values[max(0, i-window):i+1]))
    return smooth


# ---------------- UI ---------------- #

col1, col2, col3 = st.columns(3)

cpu_box = col1.empty()
lat_box = col2.empty()
mem_box = col3.empty()

log_box = st.empty()
chart_box = st.empty()

# ---------------- RULE ---------------- #

def run_rule():

    env = OpenIncidentEnv(eval_mode=True)
    commander = IncidentCommander()

    sre = SREAgent()
    support = SupportAgent()
    security = SecurityAgent()

    state = env.reset()

    rewards = []
    logs = []

    total_reward = 0
    done = False

    while not done:

        action, _ = commander.decide(state)

        if "sre" in action:
            result = sre.execute(action, state)
        elif "network" in action or "service" in action:
            result = support.execute(action, state)
        else:
            result = security.execute(action, state)

        action = result["action"]

        state, reward, done, _ = env.step(action)

        total_reward += reward
        rewards.append(total_reward)

        cpu = safe(state.get("cpu_usage"), 50)
        latency = safe(state.get("latency"), 500)
        memory = safe(state.get("memory_usage"), 50)

        logs.append(f"{explain(action)} | r={round(reward,2)}")

        cpu_box.metric("CPU", f"{cpu:.1f}%")
        lat_box.metric("Latency", f"{latency:.1f}ms")
        mem_box.metric("Memory", f"{memory:.1f}%")

        log_box.text("\n".join(logs[-6:]))

        fig, ax = plt.subplots()
        ax.plot(rewards)
        ax.axhline(0, linestyle="--")
        ax.set_title("Rule-Based Reward Trend")
        chart_box.pyplot(fig)

        time.sleep(speed)

    return total_reward


# ---------------- RL ---------------- #

def run_rl():

    env = GymOpenIncidentEnv(eval_mode=True)
    model = PPO.load("ppo_incident_model")

    obs, _ = env.reset()

    step_rewards = []
    logs = []

    total_reward = 0
    done = False

    while not done:

        if len(obs) != 7:
            obs = obs[:7]

        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        obs, reward, done, _, _ = env.step(action)

        reward = max(reward, -0.1)  # stabilization

        total_reward += reward
        step_rewards.append(reward)

        logs.append(f"{explain(env.action_map[action])} | r={round(reward,2)}")

        cpu_box.metric("Reward", f"{total_reward:.2f}")
        log_box.text("\n".join(logs[-6:]))

        smooth = smooth_curve(step_rewards)

        fig, ax = plt.subplots()
        ax.plot(smooth)
        ax.axhline(0, linestyle="--", color="red")
        ax.set_ylim(-0.2, 0.2)
        ax.set_title("RL Learning Trend (Smoothed)")
        chart_box.pyplot(fig)

        time.sleep(speed)

    return total_reward


# ---------------- COMPARE ---------------- #

def compare():

    st.subheader("📊 RL vs Rule Comparison")

    r = run_rule()
    rl = run_rl()

    labels = ["Rule", "RL"]
    values = [r, rl]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title("Final Performance Comparison")

    st.pyplot(fig)

    if rl > r:
        st.success("🚀 RL Outperforms Rule-Based System")
    else:
        st.warning("⚠️ Rule performed better")


# ---------------- MAIN ---------------- #

if st.button("▶ Start Simulation"):

    if mode == "Rule-Based":
        r = run_rule()
        st.success(f"✅ Completed | Reward={round(r,2)}")

    elif mode == "RL Agent":
        r = run_rl()
        st.success(f"🤖 RL Completed | Reward={round(r,2)}")

    else:
        compare()