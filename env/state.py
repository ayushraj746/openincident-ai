from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class State:
    # ---------------- CORE METRICS ---------------- #
    cpu_usage: float
    latency: float
    service_health: str
    memory_usage: float
    network_status: str

    # ---------------- INCIDENT CONTEXT ---------------- #
    incident_type: str = "unknown"
    severity: str = "low"

    # ---------------- MULTI-AGENT CONTEXT ---------------- #
    available_agents: List[str] = field(default_factory=lambda: ["sre", "network", "support"])

    # 🔥 NEW: agent states (realism)
    agent_status: Dict[str, str] = field(default_factory=lambda: {
        "sre": "idle",
        "network": "idle",
        "support": "idle"
    })

    # 🔥 NEW: agent reliability (used in env)
    agent_reliability: Dict[str, float] = field(default_factory=lambda: {
        "sre": 0.85,
        "network": 0.8,
        "support": 0.75
    })

    # ---------------- TEMPORAL CONTEXT ---------------- #
    step_count: int = 0

    # 🔥 NEW: action history (last few actions)
    action_history: List[str] = field(default_factory=list)

    # ---------------- RISK MODELING ---------------- #

    # 🔥 NEW: anomaly score
    anomaly_score: float = 0.0

    # 🔥 NEW: cascading failure risk
    failure_risk: float = 0.0

    # ---------------- SYSTEM DEPENDENCIES ---------------- #

    # 🔥 NEW: dependency flags
    depends_on_network: bool = True
    depends_on_memory: bool = True

    # ---------------- CONVERSION ---------------- #

    def to_dict(self):
        return {
            "cpu_usage": self.cpu_usage,
            "latency": self.latency,
            "service_health": self.service_health,
            "memory_usage": self.memory_usage,
            "network_status": self.network_status,

            "incident_type": self.incident_type,
            "severity": self.severity,

            "available_agents": self.available_agents,
            "agent_status": self.agent_status,
            "agent_reliability": self.agent_reliability,

            "step_count": self.step_count,
            "action_history": self.action_history,

            "anomaly_score": self.anomaly_score,
            "failure_risk": self.failure_risk,

            "depends_on_network": self.depends_on_network,
            "depends_on_memory": self.depends_on_memory,
        }