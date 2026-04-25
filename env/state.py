from dataclasses import dataclass, field
from typing import List


@dataclass
class State:
    # Core system metrics
    cpu_usage: float
    latency: float
    service_health: str
    memory_usage: float
    network_status: str

    # 🔥 NEW: Incident context
    incident_type: str = "unknown"
    severity: str = "low"

    # 🔥 NEW: Multi-agent context
    available_agents: List[str] = field(default_factory=lambda: ["sre", "network", "support"])

    # 🔥 NEW: Step tracking
    step_count: int = 0

    def to_dict(self):
        return {
            "cpu_usage": self.cpu_usage,
            "latency": self.latency,
            "service_health": self.service_health,
            "memory_usage": self.memory_usage,
            "network_status": self.network_status,

            # new fields
            "incident_type": self.incident_type,
            "severity": self.severity,
            "available_agents": self.available_agents,
            "step_count": self.step_count,
        }