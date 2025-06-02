# mcps_core/services/action_dispatch.py
# Service for dispatching actions from MARL agents and handling human oversight commands.

import asyncio
import json
import logging
import random
import time
import uuid # For generating unique IDs
from typing import Any, Dict, Optional

# Absolute imports for mcps package
from mcps.data_models import (
    ActionCommandModel,
    ActionResultModel,
    InterAgentMessageModel,
    MCPSAgentTypeEnum, # Using MCPS specific Enum
    HumanOversightPriorityRequestModel # From data_models.py
)
from mcps.communication_bus import IMessageBus
from mcps.context_repository import IContextRepository
from mcps.config_loader import MCPSConfigurationService

# Configure logger for this module
logger = logging.getLogger("mcps_core.services.action_dispatch")

class ActionDispatchService:
    """
    Dispatches actions commanded by MARL agents to the appropriate platform actuators.
    These actions are typically targeted at managing the MCPS platform itself or
    the services/applications it oversees (like your voice agent).
    """
    def __init__(self,
                 config_service: MCPSConfigurationService,
                 context_repo: IContextRepository,
                 message_bus: IMessageBus):
        self.config_service = config_service
        self.context_repo = context_repo
        self.message_bus = message_bus
        # In a real system, you'd initialize clients for Kubernetes, Cloud APIs, CI/CD tools, etc.
        # self.kubernetes_client = KubernetesClient() # Example
        # self.cloud_provider_client = CloudProviderClient() # Example
        logger.info("ActionDispatchService initialized.")

    async def dispatch_action(self, command: ActionCommandModel) -> ActionResultModel:
        """
        Receives an ActionCommandModel from a MARL agent and attempts to execute it.
        """
        start_time_mono = time.monotonic() # For more precise duration measurement
        logger.info(f"Dispatching action: '{command.action_type}' (ID: {command.command_id}) "
                    f"for agent {command.agent_id} ({command.agent_type.value}) "
                    f"targeting '{command.target_entity or 'platform'}'.")

        status = "PENDING"
        details: Optional[Dict[str, Any]] = None
        error_msg: Optional[str] = None

        try:
            # --- Action Handling Logic ---
            # This section simulates executing actions. In a real system,
            # this would involve calls to Kubernetes APIs, cloud provider SDKs,
            # internal platform APIs, etc.

            if command.action_type == "SCALE_SERVICE":
                # Example Parameters: service_name, desired_replicas, namespace
                service_name = command.parameters.get("service_name")
                replicas = command.parameters.get("replicas")
                if not service_name or not isinstance(replicas, int):
                    status = "INVALID_PARAMS"
                    error_msg = "Missing or invalid 'service_name' or 'replicas' for SCALE_SERVICE."
                else:
                    logger.info(f"Simulating K8s API call to scale '{service_name}' to {replicas} replicas.")
                    await asyncio.sleep(random.uniform(0.5, 2.0)) # Simulate API call latency
                    if random.random() < 0.95: # 95% success rate
                        status = "SUCCESS"
                        details = {"message": f"Service '{service_name}' scaled to {replicas} replicas.",
                                   "actual_replicas": replicas, # In real scenario, confirm actual state
                                   "target_entity": command.target_entity or service_name}
                    else:
                        status = "FAILURE"
                        error_msg = f"Simulated K8s API error during scaling of '{service_name}'."

            elif command.action_type == "TRIGGER_SECURITY_SCAN":
                # Example Parameters: scan_target (e.g., "all_services", "voice_agent_prod"), scan_profile ("quick", "full")
                scan_target = command.parameters.get("scan_target", command.target_entity or "platform_wide")
                scan_profile = command.parameters.get("scan_profile", "quick")
                logger.info(f"Simulating trigger of security scan '{scan_profile}' on target '{scan_target}'.")
                await asyncio.sleep(random.uniform(1.0, 5.0)) # Simulate scan initiation
                status = "PENDING" # Security scans are often long-running
                details = {"message": f"Security scan '{scan_profile}' initiated for '{scan_target}'.",
                           "scan_id": f"scan_{uuid.uuid4().hex[:8]}"}

            elif command.action_type == "UPDATE_PLATFORM_CONFIG":
                # Example Parameters: config_section_path, new_config_data
                section_path = command.parameters.get("config_section_path")
                new_data = command.parameters.get("new_config_data")
                if not section_path or not isinstance(new_data, dict):
                    status = "INVALID_PARAMS"
                    error_msg = "Missing or invalid 'config_section_path' or 'new_config_data'."
                else:
                    logger.info(f"Simulating update of MCPS config section '{section_path}'.")
                    # In a real system, this might call self.config_service.update_section_config()
                    # and then potentially trigger restarts or reloads of affected components.
                    await asyncio.sleep(random.uniform(0.1, 0.5))
                    # For simulation, assume direct update success
                    update_success = await self.config_service.update_section_config(section_path, new_data)
                    if update_success:
                        status = "SUCCESS"
                        details = {"message": f"MCPS config section '{section_path}' updated."}
                    else:
                        status = "FAILURE"
                        error_msg = f"Failed to update MCPS config section '{section_path}'."
            
            elif command.action_type == "RESTART_MANAGED_SERVICE":
                service_name = command.parameters.get("service_name", command.target_entity)
                if not service_name:
                    status = "INVALID_PARAMS"
                    error_msg = "Missing 'service_name' or 'target_entity' for RESTART_MANAGED_SERVICE."
                else:
                    logger.info(f"Simulating restart of managed service: '{service_name}'.")
                    await asyncio.sleep(random.uniform(1.0, 3.0))
                    status = "SUCCESS"
                    details = {"message": f"Managed service '{service_name}' restart initiated."}

            else:
                status = "UNKNOWN_ACTION"
                error_msg = f"Unknown or unsupported action type: '{command.action_type}'"
                logger.warning(error_msg)

        except Exception as e:
            logger.error(f"Error executing action {command.command_id} ('{command.action_type}'): {e}", exc_info=True)
            status = "SYSTEM_ERROR" # An error within the ActionDispatchService itself
            error_msg = str(e)

        execution_time_ms = (time.monotonic() - start_time_mono) * 1000
        action_result = ActionResultModel(
            command_id=command.command_id,
            status=status,
            details=details,
            error=error_msg,
            execution_time_ms=round(execution_time_ms, 2)
            # timestamp is auto-generated by Pydantic model
        )

        # Persist action result for auditing and agent feedback
        result_key = f"mcps_action_results:{action_result.command_id}"
        await self.context_repo.store_context_data(
            result_key,
            action_result.model_dump(mode='json'), # Use model_dump for Pydantic v2+
            ttl_seconds=self.config_service.get("context_repository.action_result_ttl_seconds", 3600 * 24) # Keep for 24 hours
        )

        # Notify the originating MARL agent about the action result
        # This is typically done via a direct message on the message bus.
        feedback_message = InterAgentMessageModel(
            sender_id="ActionDispatchService", # MCPS service ID
            recipient_id=command.agent_id, # Target the MARL agent that sent the command
            payload={"type": "action_feedback", "result": action_result.model_dump(mode='json')}
        )
        try:
            await self.message_bus.send_direct_message(command.agent_id, feedback_message)
        except Exception as e:
            logger.error(f"Failed to send action feedback to agent {command.agent_id} for command {command.command_id}: {e}", exc_info=True)
        
        logger.info(f"Action {action_result.command_id} ('{command.action_type}') completed with status: {action_result.status}")
        return action_result


class HumanOversightService:
    """
    Handles commands and directives from human operators for the MCPS platform.
    """
    def __init__(self,
                 config_service: MCPSConfigurationService,
                 message_bus: IMessageBus,
                 context_repo: IContextRepository):
        self.config_service = config_service
        self.message_bus = message_bus
        self.context_repo = context_repo
        # Default MARL agent type to target for global directives if not specified
        self.default_ops_commander_type = MCPSAgentTypeEnum.OPS_COMMANDER
        logger.info("HumanOversightService initialized.")

    async def set_global_priority(self, request: HumanOversightPriorityRequestModel) -> Dict[str, str]:
        """
        Sets a global or agent-specific priority directive, often influencing MARL agent behavior.
        """
        logger.info(f"Human operator setting priority directive: '{request.priority_level}', "
                    f"TargetAgentType: {request.target_agent_type}, TargetAgentID: {request.target_agent_id}, "
                    f"Details: {request.details}")
        
        directive_payload = {
            "directive_type": "SET_PRIORITY",
            "priority_level": request.priority_level,
            "details": request.details,
            "set_by": "human_operator_api", # Could include actual operator ID if auth is in place
            "timestamp": time.time()
        }
        
        # Store this directive in context repo for audit and agent reference
        directive_id = f"directive_priority_{uuid.uuid4().hex[:8]}"
        await self.context_repo.store_context_data(
            f"mcps_human_directives:{directive_id}",
            directive_payload,
            ttl_seconds=self.config_service.get("context_repository.directive_ttl_seconds", 3600 * 24 * 7) # Keep for 7 days
        )

        # Prepare message for the bus
        message_to_send = InterAgentMessageModel(
            sender_id="HumanOversightService", # From this MCPS service
            payload=directive_payload
        )

        target_info = ""
        if request.target_agent_id:
            message_to_send.recipient_id = request.target_agent_id
            await self.message_bus.send_direct_message(request.target_agent_id, message_to_send)
            target_info = f"specific agent '{request.target_agent_id}'"
        elif request.target_agent_type:
            # Message all agents of a specific type via a dedicated topic
            target_topic = f"mcps_directives_type_{request.target_agent_type.value}"
            message_to_send.topic = target_topic
            await self.message_bus.publish(target_topic, message_to_send)
            target_info = f"all '{request.target_agent_type.value}' agents via topic '{target_topic}'"
        else:
            # Default to a general OpsCommander topic or a platform-wide directives topic
            default_target_topic = f"mcps_directives_type_{self.default_ops_commander_type.value}"
            message_to_send.topic = default_target_topic
            await self.message_bus.publish(default_target_topic, message_to_send)
            target_info = f"default target '{self.default_ops_commander_type.value}' agents via topic '{default_target_topic}'"
            
        return {"status": "success", "message": f"Priority directive '{request.priority_level}' dispatched to {target_info}."}

    async def emergency_shutdown_signal(self, reason: str, initiated_by: str = "human_operator_api") -> Dict[str, str]:
        """
        Broadcasts an emergency shutdown signal across the MCPS platform.
        MARL agents and critical services should subscribe to this.
        """
        logger.warning(f"EMERGENCY SHUTDOWN SIGNAL initiated by '{initiated_by}'. Reason: '{reason}'")
        
        shutdown_payload = {
            "directive_type": "EMERGENCY_SHUTDOWN",
            "reason": reason,
            "initiated_by": initiated_by,
            "timestamp": time.time()
        }
        # Store for audit
        shutdown_id = f"emergency_shutdown_{uuid.uuid4().hex[:8]}"
        await self.context_repo.store_context_data(
            f"mcps_human_directives:{shutdown_id}",
            shutdown_payload,
            ttl_seconds=self.config_service.get("context_repository.critical_event_ttl_seconds", 3600 * 24 * 30) # Keep for 30 days
        )

        # Publish to a well-known, high-priority emergency topic
        emergency_topic = self.config_service.get("message_bus.emergency_topic", "mcps_platform_emergency_commands")
        message = InterAgentMessageModel(
            sender_id="HumanOversightService",
            topic=emergency_topic,
            payload=shutdown_payload,
            priority=10 # Highest priority
        )
        await self.message_bus.publish(emergency_topic, message)
        
        return {"status": "success", "message": f"Emergency shutdown signal broadcasted to topic '{emergency_topic}'."}

if __name__ == "__main__":
    # Example Usage (requires mcps_core.data_models, etc.)
    logging.basicConfig(level=logging.DEBUG)

    # Mock dependencies
    class MockMCPSConfigService:
        def get(self, key_path, default=None):
            if key_path == "context_repository.action_result_ttl_seconds": return 3600
            if key_path == "context_repository.directive_ttl_seconds": return 3600 * 24
            if key_path == "message_bus.emergency_topic": return "test_emergency_topic"
            return default

    class MockContextRepo(IContextRepository):
        async def connect(self): pass
        async def disconnect(self): pass
        async def store_context_data(self, key, data, ttl_seconds=None): print(f"MockCtxRepo STORE: {key} = {str(data)[:60]}..."); return True
        async def get_context_data(self, key): return None
        async def delete_context_data(self, key): return True
        async def get_time_series_data(self, m, s, e, a=None): return []
        async def append_to_list(self, k, v, ml=None): return 0
        async def get_list_range(self, k, s, e): return []

    class MockMessageBus(IMessageBus):
        async def connect(self): pass
        async def disconnect(self): pass
        async def publish(self, topic, message): print(f"MockMsgBus PUBLISH Topic '{topic}': {message.payload}")
        async def subscribe(self, topic, callback): return "mock_sub_id"
        async def unsubscribe(self, sub_id): pass
        async def send_direct_message(self, recipient_id, message): print(f"MockMsgBus DIRECT to '{recipient_id}': {message.payload}")

    async def demo_dispatch_and_oversight():
        config = MockMCPSConfigService()
        repo = MockContextRepo()
        bus = MockMessageBus()

        action_service = ActionDispatchService(config, repo, bus)
        oversight_service = HumanOversightService(config, repo, bus)

        # Test Action Dispatch
        print("\n--- Testing Action Dispatch ---")
        action_cmd = ActionCommandModel(
            agent_id="ScaleGuardian_001",
            agent_type=MCPSAgentTypeEnum.SCALE_GUARDIAN,
            action_type="SCALE_SERVICE",
            parameters={"service_name": "voice_agent_core_prod", "replicas": 5},
            target_entity="voice_agent_core_prod"
        )
        result = await action_service.dispatch_action(action_cmd)
        print(f"Action Result: {result.model_dump_json(indent=2)}")

        unknown_action_cmd = ActionCommandModel(
            agent_id="TestAgent_002",
            agent_type=MCPSAgentTypeEnum.GENERIC_MCPS_AGENT,
            action_type="DO_NONEXISTENT_THING",
            parameters={}
        )
        result_unknown = await action_service.dispatch_action(unknown_action_cmd)
        print(f"Unknown Action Result: {result_unknown.model_dump_json(indent=2)}")


        # Test Human Oversight
        print("\n--- Testing Human Oversight ---")
        priority_req = HumanOversightPriorityRequestModel(
            priority_level="MAX_PERFORMANCE_MODE",
            details={"reason": "Peak business hours approaching"},
            target_agent_type=MCPSAgentTypeEnum.SCALE_GUARDIAN
        )
        oversight_res1 = await oversight_service.set_global_priority(priority_req)
        print(f"Set Priority Result: {oversight_res1}")

        oversight_res2 = await oversight_service.emergency_shutdown_signal(reason="Unforeseen critical system anomaly.")
        print(f"Emergency Shutdown Result: {oversight_res2}")

    # asyncio.run(demo_dispatch_and_oversight()) # Comment out if not testing directly
