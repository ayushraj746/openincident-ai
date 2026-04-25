import random
from .state import State


def generate_task(difficulty: str) -> State:

    # 🔥 Incident types
    incident_types = [
        "cpu_spike",
        "memory_leak",
        "network_failure",
        "deployment_failure",
    ]

    incident = random.choice(incident_types)

    # ---------------- EASY ---------------- #
    if difficulty == "easy":
        return State(
            cpu_usage=random.randint(40, 60),
            latency=random.randint(200, 400),
            service_health="degraded",
            memory_usage=random.randint(40, 60),
            network_status="normal",

            incident_type=incident,
            severity="low",
            available_agents=["sre"],
        )

    # ---------------- MEDIUM ---------------- #
    elif difficulty == "medium":
        cpu = random.randint(70, 90)
        latency = random.randint(600, 1000)
        memory = random.randint(60, 85)
        network = random.choice(["normal", "slow"])
        service_state = random.choice(["degraded", "down"])

        return State(
            cpu_usage=cpu,
            latency=latency,
            service_health=service_state,
            memory_usage=memory,
            network_status=network,

            incident_type=incident,
            severity="medium",
            available_agents=["sre", "network"],
        )

    # ---------------- HARD ---------------- #
    elif difficulty == "hard":
        cpu = random.randint(85, 100)
        latency = random.randint(1200, 1800)
        memory = random.randint(75, 95)
        network = random.choice(["slow", "down"])

        return State(
            cpu_usage=cpu,
            latency=latency,
            service_health="down",
            memory_usage=memory,
            network_status=network,

            incident_type=incident,
            severity="high",
            available_agents=["sre", "network", "support"],
        )

    else:
        raise ValueError("Invalid difficulty")