import streamlit as st
import time
import matplotlib.pyplot as plt

from env.environment import OpenIncidentEnv
from env.gym_wrapper import GymOpenIncidentEnv
from stable_baselines3 import PPO
from agents.commander import IncidentCommander
from agents.sre_agent import SREAgent
from agents.support_agent import SupportAgent
from agents.security_agent import SecurityAgent

st.set_page_config(page_title="AutoSRE AI", layout="wide")

st.title("🚀 AutoSRE AI — Incident Response System")
st.markdown("Real-time Multi-Agent AI for System Recovery")

# ---------------- SIDEBAR ---------------- #

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Rule-Based", "RL Agent", "Compare"]
)

speed = st.sidebar.slider("Simulation Speed", 0.1, 1.5, 0.5)

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


# ---------------- LIVE PANELS ---------------- #

metric_placeholder = st.empty()
log_placeholder = st.empty()
chart_placeholder = st.empty()

# ---------------- RULE SIMULATION ---------------- #

def run_rule():

    env = OpenIncidentEnv(eval_mode=True)

    commander = IncidentCommander()
    sre = SREAgent()
    support = SupportAgent()
    security = SecurityAgent()

    state = env.reset()

    logs = []
    rewards = []
    cpu_hist = []

    total_reward = 0
    step = 0
    done = False

    while not done:
        step += 1

        action, reason = commander.decide(state)

        # choose agent
        if "sre" in action:
            result = sre.execute(action, state)
        elif "network" in action or "service" in action:
            result = support.execute(action, state)
        else:
            result = security.execute(action, state)

        action = result["action"]

        state, reward, done, info = env.step(action)

        total_reward += reward
        rewards.append(total_reward)
        cpu_hist.append(safe(state.get("cpu_usage"),50))

        logs.append(f"{step}: {explain(action)} | r={reward:.2f}")

        # ---------------- LIVE UPDATE ---------------- #

        metric_placeholder.metric("CPU", f"{cpu_hist[-1]:.1f}")
        log_placeholder.text("\n".join(logs[-8:]))

        fig, ax = plt.subplots()
        ax.plot(rewards)
        ax.set_title("Reward Trend")
        chart_placeholder.pyplot(fig)

        time.sleep(speed)

    return step, total_reward


# ---------------- RL SIMULATION ---------------- #

def run_rl():

    env = GymOpenIncidentEnv(eval_mode=True)
    model = PPO.load("models/ppo_incident_model")

    obs, _ = env.reset()

    rewards = []
    logs = []

    total_reward = 0
    step = 0
    done = False

    action_map = env.action_map

    while not done:
        step += 1

        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        obs, reward, done, _, _ = env.step(action)

        total_reward += reward
        rewards.append(total_reward)

        logs.append(f"{step}: {explain(action_map[action])} | r={reward:.2f}")

        # ---------------- LIVE UPDATE ---------------- #

        metric_placeholder.metric("Reward", f"{total_reward:.2f}")
        log_placeholder.text("\n".join(logs[-8:]))

        fig, ax = plt.subplots()
        ax.plot(rewards)
        ax.set_title("Reward Trend")
        chart_placeholder.pyplot(fig)

        time.sleep(speed)

    return step, total_reward


# ---------------- MAIN ---------------- #

if st.button("▶ Run Simulation"):

    st.divider()

    if mode == "Rule-Based":
        steps, reward = run_rule()

        st.success("✅ Recovery Complete")
        st.metric("Steps", steps)
        st.metric("Total Reward", round(reward, 2))

    elif mode == "RL Agent":
        steps, reward = run_rl()

        st.success("🤖 RL Recovery Complete")
        st.metric("Steps", steps)
        st.metric("Total Reward", round(reward, 2))

    else:
        st.subheader("⚔️ RL vs Rule Comparison")

        r_steps, r_reward = run_rule()
        rl_steps, rl_reward = run_rl()

        col1, col2 = st.columns(2)

        col1.metric("Rule Steps", r_steps)
        col1.metric("Rule Reward", round(r_reward, 2))

        col2.metric("RL Steps", rl_steps)
        col2.metric("RL Reward", round(rl_reward, 2))

        if rl_steps < r_steps:
            st.success("🚀 RL is faster")
        else:
            st.warning("⚠️ Rule-based performed better")