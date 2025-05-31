# mcps_core/data_models.py
# Pydantic models for API validation and internal data structures for MCPS.

import uuid
import time
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from enum import Enum

# Agent types for the MARL agents *managing* the MCPS platform
class MCPSAgentTypeEnum(str, Enum):
    SCALE_GUARDIAN = "ScaleGuardian"
    SECOPS_SENTINEL = "SecOpsSentinel"
    DEPLOY_MASTER = "DeployMaster"
    OPS_COMMANDER = "OpsCommander" # Example: A higher-level orchestrator agent
    COST_OPTIMIZER = "CostOptimizer"
    DATA_PATRON = "DataPatron" # Manages data aspects of the platform
    SYNAPSE_TUNER = "SynapseTuner" # Optimizes internal AI models of MCPS
    GENERIC_MCPS_AGENT = "GenericMCPSAgent" # Fallback or general purpose

class ObservationModel(BaseModel):
    """Observation data provided TO a MARL agent managing MCPS."""
    agent_type: MCPSAgentTypeEnum
    agent_id: str # The ID of the MARL agent receiving the observation
    timestamp: float = Field(default_factory=time.time)
    data: Dict[str, Any] # Context data relevant to this MCPS agent's function
    context_version: str = "1.0"

class ActionCommandModel(BaseModel):
    """Action command FROM a MARL agent to be executed BY MCPS on the platform it manages."""
    agent_id: str # The ID of the MARL agent issuing the command
    agent_type: MCPSAgentTypeEnum
    action_type: str # e.g., "SCALE_SERVICE", "RUN_SECURITY_SCAN", "UPDATE_AGENT_CONFIG"
    parameters: Dict[str, Any]
    target_entity: Optional[str] = None # e.g., "voice_agent_service_prod", "redis_cluster_main"
    command_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = Field(default=5, ge=1, le=10) # 1=lowest, 10=highest

class ActionResultModel(BaseModel):
    """Result of an action executed BY MCPS."""
    command_id: str
    status: str # e.g., "SUCCESS", "FAILURE", "PENDING", "INVALID_PARAMS"
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None
    timestamp: float = Field(default_factory=time.time)

class InterAgentMessageModel(BaseModel):
    """For communication between MCPS platform services or MARL agents via the message bus."""
    sender_id: str # Could be an agent_id or an mcps_service_name
    recipient_id: Optional[str] = None # For direct messaging
    topic: Optional[str] = None # For pub/sub
    payload: Dict[str, Any]
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    priority: int = Field(default=5, ge=1, le=10)

    @validator('recipient_id', 'topic', always=True)
    def check_recipient_or_topic_provided(cls, v, values):
        if values.get('recipient_id') is None and values.get('topic') is None:
            raise ValueError('Either recipient_id or topic must be provided')
        if values.get('recipient_id') is not None and values.get('topic') is not None:
            raise ValueError('Provide either recipient_id or topic, not both')
        return v

class MCPToolDefinitionModel(BaseModel): # Renamed from MCPToolModel to avoid confusion
    """Definition of a tool exposed by the MCPS for other systems or agents."""
    name: str
    description: str
    input_schema: Dict[str, Any] = Field(alias="inputSchema")

class MCPToolCallRequestModel(BaseModel):
    """Request to call an MCPS tool."""
    name: str
    arguments: Dict[str, Any]

class MCPToolCallResponseModel(BaseModel):
    """Response from an MCPS tool call."""
    tool_name: str
    status: str # "SUCCESS" or "ERROR"
    result: Optional[Any] = None
    error: Optional[str] = None

class HumanOversightPriorityRequestModel(BaseModel):
    """Request model for human operator to set priorities for MCPS MARL agents."""
    priority_level: str # e.g., "FOCUS_ON_COST_REDUCTION", "MAXIMIZE_STABILITY"
    details: Dict[str, Any]
    target_agent_type: Optional[MCPSAgentTypeEnum] = None
    target_agent_id: Optional[str] = None # Specific MARL agent ID

class ConfigSectionUpdateRequestModel(BaseModel):
    """Request model for updating a section of the MCPS configuration."""
    data: Dict[str, Any] # The new data for the section

class HealthStatusModel(BaseModel):
    """Overall health status of the MCPS platform."""
    status: str # "HEALTHY", "DEGRADED", "UNHEALTHY"
    timestamp: float = Field(default_factory=time.time)
    services: Optional[Dict[str, str]] = None # Status of individual MCPS internal services

# Models for Tenancy (simplified from mcp_tenant_architecture.py for now)
class TenantModel(BaseModel):
    id: Optional[uuid.UUID] = None
    name: str
    subdomain: str
    tier: str # e.g., "starter", "enterprise"
    status: str = "active"

# Models for Security Audits (simplified from mcp_securit_audit.py)
class SecurityAuditReportSummaryModel(BaseModel):
    report_id: str
    timestamp: float
    overall_security_score: float
    critical_vulnerabilities: int
    high_vulnerabilities: int

# Add other Pydantic models as needed from your various modules like
# mcp_integration.py (e.g., for specific metric structures if they pass through API)
# mcp_tenant_architecture.py (for API interactions with tenant data)
