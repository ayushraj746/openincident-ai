import streamlit as st
from env.environment import OpenIncidentEnv
from stable_baselines3 import PPO
from env.gym_wrapper import GymOpenIncidentEnv

st.set_page_config(page_title="OpenIncident AI", layout="wide")

st.title("🚀 OpenIncident AI Agent Demo")
st.markdown("### Intelligent Incident Recovery using RL vs Rule-Based Systems")

mode = st.selectbox("Select Mode", ["Rule-Based", "RL Agent"])


# ---------------- SAFE VALUE HANDLING ---------------- #
def safe(val, default):
    return val if val is not None else default


# ---------------- ACTION EXPLAINER ---------------- #
def explain(action):
    mapping = {
        "restart_service": "🔧 Restarting failed service",
        "scale_resources": "📈 Scaling compute resources",
        "restart_network": "🌐 Fixing network issues",
        "clear_cache": "🧹 Clearing memory cache",
        "do_nothing": "⏳ System stabilizing",
    }
    return mapping.get(action, action)


# ---------------- STATUS DISPLAY ---------------- #
def show_status(state):
    col1, col2, col3, col4, col5 = st.columns(5)

    cpu = safe(state.get("cpu_usage"), 50)
    latency = safe(state.get("latency"), 500)
    memory = safe(state.get("memory_usage"), 50)

    col1.metric("CPU", f"{cpu:.1f}")
    col2.metric("Latency", f"{latency:.1f}")
    col3.metric("Memory", f"{memory:.1f}")
    col4.metric("Service", state.get("service_health", "unknown"))
    col5.metric("Network", state.get("network_status", "unknown"))


# ---------------- RUN BUTTON ---------------- #
if st.button("▶️ Run Simulation"):

    st.divider()

    # -------- RULE-BASED -------- #
    if mode == "Rule-Based":
        from agents.commander import IncidentCommander
        from agents.sre_agent import SREAgent
        from agents.support_agent import SupportAgent

        env = OpenIncidentEnv(eval_mode=True)

        commander = IncidentCommander()
        sre = SREAgent()
        support = SupportAgent()

        state = env.reset()

        st.subheader("📊 Initial System State")
        show_status(state)

        step = 0
        done = False

        progress_bar = st.progress(0)
        log_area = st.empty()

        logs = ""

        while not done:
            step += 1

            agent_name, action = commander.decide(state)

            if agent_name == "sre":
                action = sre.execute(action)
            else:
                action = support.execute(action)

            state, reward, done, _ = env.step(action)

            logs += f"Step {step} → {explain(action)} | Reward: {reward:.2f}\n"
            log_area.text(logs)

            progress_bar.progress(min(step / 15, 1.0))

            show_status(state)

        st.success("✅ System Recovered Successfully!")

    # -------- RL AGENT -------- #
    else:
        env = GymOpenIncidentEnv(eval_mode=True)
        model = PPO.load("ppo_incident_model")

        obs, _ = env.reset()

        st.subheader("📊 RL Agent Running (Normalized State Space)")

        step = 0
        done = False

        progress_bar = st.progress(0)
        log_area = st.empty()

        logs = ""

        action_map = {
            0: "scale_resources",
            1: "restart_service",
            2: "clear_cache",
            3: "restart_network",
            4: "do_nothing",
        }

        while not done:
            step += 1

            action, _ = model.predict(obs)
            action = int(action)

            obs, reward, done, _, _ = env.step(action)

            action_name = action_map[action]

            logs += f"Step {step} → {explain(action_name)} | Reward: {reward:.2f}\n"
            log_area.text(logs)

            progress_bar.progress(min(step / 15, 1.0))

        st.success("🤖 RL Agent Successfully Recovered the System!")