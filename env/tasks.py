import random
from .state import State


def generate_task(difficulty: str, step_count: int = 0) -> State:

    # ---------------- INCIDENT TYPES ---------------- #
    incident_types = [
        "cpu_spike",
        "memory_leak",
        "network_failure",
        "deployment_failure",
        "cascading_failure",   # 🔥 NEW
    ]

    # ---------------- DYNAMIC DIFFICULTY ---------------- #
    if step_count > 15:
        difficulty = "hard"
    elif step_count > 8:
        difficulty = "medium"

    incident = random.choice(incident_types)

    # ---------------- BASE METRICS ---------------- #

    cpu = random.randint(40, 60)
    latency = random.randint(200, 400)
    memory = random.randint(40, 60)
    network = "normal"
    service = "healthy"

    anomaly_score = random.uniform(0.1, 0.3)
    failure_risk = random.uniform(0.1, 0.3)

    # ---------------- INCIDENT EFFECTS ---------------- #

    if incident == "cpu_spike":
        cpu += random.randint(30, 50)
        latency += random.randint(200, 500)

    elif incident == "memory_leak":
        memory += random.randint(30, 50)
        latency += random.randint(100, 300)

    elif incident == "network_failure":
        network = random.choice(["slow", "down"])
        latency += random.randint(300, 800)

    elif incident == "deployment_failure":
        service = "degraded"
        cpu += random.randint(10, 30)
        latency += random.randint(300, 700)

    elif incident == "cascading_failure":
        cpu += random.randint(30, 60)
        memory += random.randint(30, 60)
        latency += random.randint(500, 1000)
        network = random.choice(["slow", "down"])
        service = "down"
        anomaly_score = random.uniform(0.6, 1.0)
        failure_risk = random.uniform(0.6, 1.0)

    # ---------------- DIFFICULTY ADJUSTMENTS ---------------- #

    if difficulty == "easy":
        severity = "low"
        available_agents = ["sre"]

    elif difficulty == "medium":
        severity = "medium"
        cpu += random.randint(10, 20)
        memory += random.randint(10, 20)
        latency += random.randint(200, 400)
        available_agents = ["sre", "network"]

    elif difficulty == "hard":
        severity = "high"
        cpu += random.randint(20, 40)
        memory += random.randint(20, 40)
        latency += random.randint(400, 800)
        network = random.choice(["slow", "down"])
        service = "down"
        available_agents = ["sre", "network", "support"]

    else:
        raise ValueError("Invalid difficulty")

    # ---------------- CLAMP VALUES ---------------- #

    cpu = min(cpu, 100)
    memory = min(memory, 100)

    # ---------------- CREATE STATE ---------------- #

    state = State(
        cpu_usage=cpu,
        latency=latency,
        service_health=service,
        memory_usage=memory,
        network_status=network,

        incident_type=incident,
        severity=severity,
        available_agents=available_agents,

        anomaly_score=anomaly_score,
        failure_risk=failure_risk,
    )

    return state