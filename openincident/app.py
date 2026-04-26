import streamlit as st
import time
import matplotlib.pyplot as plt

from env.environment import OpenIncidentEnv
from env.gym_wrapper import GymOpenIncidentEnv
from stable_baselines3 import PPO

st.set_page_config(page_title="OpenIncident AI", layout="wide")

st.title("OpenIncident AI Dashboard")
st.markdown("Autonomous Incident Recovery using Rule-Based and RL Agents")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Rule-Based", "RL Agent", "Compare Both"]
)

# ---------------- HELPERS ---------------- #

def safe(val, default):
    return val if val is not None else default


def explain(action):
    mapping = {
        "restart_service": "Restarting service",
        "delegate_sre": "Scaling infrastructure",
        "delegate_memory": "Clearing memory",
        "delegate_network": "Fixing network",
        "rollback_deployment": "Rollback deployment",
        "do_nothing": "System stabilizing",
    }
    return mapping.get(action, action)


def show_status(state):
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("CPU", f"{safe(state.get('cpu_usage'),50):.1f}")
    col2.metric("Latency", f"{safe(state.get('latency'),500):.1f}")
    col3.metric("Memory", f"{safe(state.get('memory_usage'),50):.1f}")
    col4.metric("Service", state.get("service_health", "unknown"))
    col5.metric("Network", state.get("network_status", "unknown"))


# ---------------- RUN SIMULATION ---------------- #

if st.button("Run Simulation"):

    st.divider()

    def run_rule():
        from agents.commander import IncidentCommander
        from agents.sre_agent import SREAgent
        from agents.support_agent import SupportAgent

        env = OpenIncidentEnv(eval_mode=True)

        commander = IncidentCommander()
        sre = SREAgent()
        support = SupportAgent()

        state = env.reset()

        steps = 0
        total_reward = 0
        done = False

        logs = []
        rewards = []
        cpu_hist = []

        while not done:
            steps += 1

            agent, action, _ = commander.decide(state)

            if agent == "sre":
                result = sre.execute(action, state)
            else:
                result = support.execute(action, state)

            action = result["action"]

            state, reward, done, _ = env.step(action)

            total_reward += reward
            rewards.append(total_reward)
            cpu_hist.append(safe(state.get("cpu_usage"),50))

            logs.append(f"{steps}: {explain(action)} | r={reward:.2f}")

        return steps, total_reward, logs, rewards, cpu_hist


    def run_rl():
        env = GymOpenIncidentEnv(eval_mode=True)
        model = PPO.load("ppo_incident_model.zip")

        obs, _ = env.reset()

        steps = 0
        total_reward = 0
        done = False

        logs = []
        rewards = []

        action_map = {
            0: "delegate_sre",
            1: "restart_service",
            2: "delegate_memory",
            3: "delegate_network",
            4: "do_nothing",
        }

        while not done:
            steps += 1

            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            obs, reward, done, _, _ = env.step(action)

            total_reward += reward
            rewards.append(total_reward)

            logs.append(f"{steps}: {explain(action_map[action])} | r={reward:.2f}")

        return steps, total_reward, logs, rewards


    # ---------------- EXECUTION ---------------- #

    if mode == "Rule-Based":
        steps, total_reward, logs, rewards, cpu = run_rule()

        st.subheader("Rule-Based Execution")

        st.text("\n".join(logs[-10:]))

        st.success("Recovered")
        st.metric("Steps", steps)
        st.metric("Total Reward", round(total_reward,2))

        fig, ax = plt.subplots()
        ax.plot(rewards)
        ax.set_title("Reward Trend")
        st.pyplot(fig)

    elif mode == "RL Agent":
        steps, total_reward, logs, rewards = run_rl()

        st.subheader("RL Agent Execution")

        st.text("\n".join(logs[-10:]))

        st.success("Recovered")
        st.metric("Steps", steps)
        st.metric("Total Reward", round(total_reward,2))

        fig, ax = plt.subplots()
        ax.plot(rewards)
        ax.set_title("Reward Trend")
        st.pyplot(fig)

    else:
        st.subheader("Comparison Mode")

        r_steps, r_reward, _, _, _ = run_rule()
        rl_steps, rl_reward, _, _ = run_rl()

        col1, col2 = st.columns(2)

        col1.markdown("### Rule-Based")
        col1.metric("Steps", r_steps)
        col1.metric("Reward", round(r_reward,2))

        col2.markdown("### RL Agent")
        col2.metric("Steps", rl_steps)
        col2.metric("Reward", round(rl_reward,2))

        if rl_steps < r_steps:
            st.success("RL is faster")
        else:
            st.warning("Rule-Based is currently better")