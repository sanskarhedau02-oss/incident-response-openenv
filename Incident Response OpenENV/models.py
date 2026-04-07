# Author: <Sanskar Hedau>
# Project: Incident Response OpenEnv
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal


class AIAction(BaseModel):
    action_type: Literal[
        "restart_service", "scale_up", "rollback", "flush_cache", "escalate", "noop"
    ] = Field(..., description="The type of remediation action to apply.")
    target: Optional[str] = Field(
        None,
        description="Service name for restart_service actions (e.g. 'auth-service', 'api').",
    )


class AIObservation(BaseModel):
    incident_id: str
    services: Dict[str, str]
    cpu_usage: float
    memory_usage: float
    error_rate: float
    latency_ms: float
    incident_severity: int
    alert_count: int
    logs: List[str]
    step_count: int


class StepResponse(BaseModel):
    observation: AIObservation
    reward: float
    done: bool
    info: Dict


class ResetResponse(BaseModel):
    observation: AIObservation
    reward: float = 0.0
    done: bool = False
    task: str
