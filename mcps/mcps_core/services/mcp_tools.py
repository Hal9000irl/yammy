# mcps_core/services/mcp_tools.py
# MCP Tool handling logic and definitions for tools exposed BY the MCPS platform.

import logging
import time
from typing import Dict, Any, Optional, List, Callable, Awaitable

# Absolute imports for mcps package
from mcps.data_models import (
    MCPToolDefinitionModel, # Renamed from MCPToolModel in previous data_models for clarity
    MCPToolCallRequestModel,
    MCPToolCallResponseModel,
    MCPSAgentTypeEnum, # Using MCPS specific Enum
    ObservationModel, # To return structure from get_agent_observation tool
    ActionCommandModel, # To construct for execute_agent_action tool
    ActionResultModel, # To return structure from execute_agent_action tool
    HumanOversightPriorityRequestModel # To construct for set_human_priority_directive tool
)
# Import the services that the tools will interact with
from .observation import AgentObservationService
from .action_dispatch import ActionDispatchService, HumanOversightService
# Absolute import for context repository
from mcps.context_repository import IContextRepository # For tools like get_platform_status

# Configure logger for this module
logger = logging.getLogger("mcps_core.services.mcp_tools")

class MCPToolHandler:
    """
    Manages and executes tools that are exposed by the MCPS platform itself.
    These tools can be called by external systems, human operators via an API,
    or even by the MARL agents managing the MCPS if they are designed to use a tool-calling paradigm.
    """
    def __init__(self,
                 observation_service: AgentObservationService,
                 action_service: ActionDispatchService,
                 oversight_service: HumanOversightService,
                 context_repo: IContextRepository): # For tools that need direct context access
        self.observation_service = observation_service
        self.action_service = action_service
        self.oversight_service = oversight_service
        self.context_repo = context_repo

        # Define the tools available on the MCPS platform
        self.tools: Dict[str, MCPToolDefinitionModel] = {
            "get_platform_agent_observation": MCPToolDefinitionModel(
                name="get_platform_agent_observation",
                description="Retrieves the current observation data for a specified MARL agent managing the MCPS platform.",
                inputSchema={ # JSON Schema for input arguments
                    "type": "object",
                    "properties": {
                        "agent_type": {"type": "string", "enum": [e.value for e in MCPSAgentTypeEnum]},
                        "agent_id": {"type": "string", "description": "The unique ID of the MARL agent."}
                    },
                    "required": ["agent_type", "agent_id"]
                }
            ),
            "execute_platform_agent_action": MCPToolDefinitionModel(
                name="execute_platform_agent_action",
                description="Dispatches an action command for a MARL agent to be executed by the MCPS on the platform it manages.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "agent_type": {"type": "string", "enum": [e.value for e in MCPSAgentTypeEnum]},
                        "action_type": {"type": "string", "description": "e.g., SCALE_SERVICE, TRIGGER_SECURITY_SCAN"},
                        "parameters": {"type": "object", "description": "Action-specific parameters"},
                        "target_entity": {"type": "string", "nullable": True, "description": "The entity the action targets, e.g., a service name"},
                        "priority": {"type": "integer", "minimum": 1, "maximum": 10, "nullable": True, "default": 5}
                    },
                    "required": ["agent_id", "agent_type", "action_type", "parameters"]
                }
            ),
            "get_mcps_platform_status": MCPToolDefinitionModel(
                name="get_mcps_platform_status",
                description="Retrieves a summary of the current MCPS platform status, including key metrics and component health.",
                inputSchema={"type": "object", "properties": {}} # No arguments needed for this simple version
            ),
            "set_human_oversight_priority": MCPToolDefinitionModel(
                name="set_human_oversight_priority",
                description="Allows a human operator to set a global or agent-specific priority directive for MCPS MARL agents.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "priority_level": {"type": "string", "description": "e.g., HIGH_COST_SAVING, MAX_PERFORMANCE"},
                        "details": {"type": "object", "description": "Additional context or parameters for the directive"},
                        "target_agent_type": {"type": "string", "enum": [e.value for e in MCPSAgentTypeEnum], "nullable": True},
                        "target_agent_id": {"type": "string", "nullable": True}
                    },
                    "required": ["priority_level", "details"]
                }
            )
            # Add more tools as the MCPS platform evolves
        }
        
        # Map tool names to their implementation methods
        self.tool_implementations: Dict[str, Callable[[Dict[str, Any]], Awaitable[Any]]] = {
            "get_platform_agent_observation": self._handle_get_platform_agent_observation,
            "execute_platform_agent_action": self._handle_execute_platform_agent_action,
            "get_mcps_platform_status": self._handle_get_mcps_platform_status,
            "set_human_oversight_priority": self._handle_set_human_oversight_priority,
        }
        logger.info(f"MCPToolHandler initialized with {len(self.tools)} tools for the MCPS platform.")

    async def list_tools(self) -> List[MCPToolDefinitionModel]:
        """Returns a list of all defined MCP tools for the platform."""
        return list(self.tools.values())

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolCallResponseModel:
        """
        Calls the specified MCP tool with the given arguments.
        Includes basic validation (tool existence). Argument schema validation should be added.
        """
        logger.info(f"Attempting to call MCPS platform tool: '{tool_name}' with arguments: {arguments}")
        
        if tool_name not in self.tool_implementations:
            logger.error(f"MCPS Tool '{tool_name}' not found.")
            return MCPToolCallResponseModel(tool_name=tool_name, status="ERROR", error="Tool not found")
        
        # TODO: Implement JSON Schema validation for 'arguments' against self.tools[tool_name].inputSchema
        # For example, using a library like 'jsonschema'.
        # try:
        #     validate(instance=arguments, schema=self.tools[tool_name].inputSchema)
        # except ValidationError as ve:
        #     logger.warning(f"Invalid arguments for tool '{tool_name}': {ve.message}")
        #     return MCPToolCallResponseModel(tool_name=tool_name, status="ERROR", error=f"Invalid arguments: {ve.message}")

        try:
            implementation_method = self.tool_implementations[tool_name]
            # The implementation method is expected to return the data that goes into the 'result' field.
            result_data = await implementation_method(arguments)
            return MCPToolCallResponseModel(tool_name=tool_name, status="SUCCESS", result=result_data)
        except Exception as e:
            logger.error(f"Error executing MCPS tool '{tool_name}': {e}", exc_info=True)
            return MCPToolCallResponseModel(tool_name=tool_name, status="ERROR", error=str(e))

    # --- Tool Implementation Methods ---

    async def _handle_get_platform_agent_observation(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handles the 'get_platform_agent_observation' tool call."""
        # Arguments should have been validated by Pydantic if called via FastAPI endpoint
        # or by JSON schema validation if called directly.
        try:
            agent_type_val = args["agent_type"]
            # Ensure agent_type_val is an instance of MCPSAgentTypeEnum or can be converted
            if not isinstance(agent_type_val, MCPSAgentTypeEnum):
                 agent_type = MCPSAgentTypeEnum(agent_type_val) # May raise ValueError if invalid
            else:
                agent_type = agent_type_val

            agent_id = args["agent_id"]
        except KeyError as ke:
            raise ValueError(f"Missing required argument for get_platform_agent_observation: {ke}")
        except ValueError as ve: # For invalid enum value
            raise ValueError(f"Invalid agent_type for get_platform_agent_observation: {args['agent_type']}. Valid types: {[e.value for e in MCPSAgentTypeEnum]}") from ve

        observation_model: ObservationModel = await self.observation_service.get_observation_for_agent(agent_type, agent_id)
        return observation_model.model_dump(mode='json') # Return as dict

    async def _handle_execute_platform_agent_action(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handles the 'execute_platform_agent_action' tool call."""
        try:
            # Construct the ActionCommandModel from arguments. Pydantic handles validation.
            command = ActionCommandModel(**args)
        except Exception as pydantic_error: # Catch Pydantic validation errors
            raise ValueError(f"Invalid arguments for execute_platform_agent_action: {pydantic_error}") from pydantic_error
            
        action_result_model: ActionResultModel = await self.action_service.dispatch_action(command)
        return action_result_model.model_dump(mode='json')

    async def _handle_get_mcps_platform_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handles the 'get_mcps_platform_status' tool call."""
        # This is a simplified status. A real implementation would aggregate more data.
        # It could call various internal health checks or query specific context keys.
        logger.debug("Fetching MCPS platform status via tool call.")
        
        # Example: Fetch some key metrics from the context repository
        metrics_overview = await self.context_repo.get_context_data("platform_metrics:cluster_summary")
        k8s_sim_status = await self.context_repo.get_context_data("kubernetes_api:cluster_status_simulation")
        active_threats_summary = await self.context_repo.get_context_data("active_threats:summary") # Hypothetical key

        # Overall health could be determined by a more sophisticated HealthChecker service
        overall_health = "OPERATIONAL" # Default
        if not metrics_overview or not k8s_sim_status:
            overall_health = "DEGRADED (Missing key metrics or K8s status)"
        elif active_threats_summary and active_threats_summary.get("critical_threat_count", 0) > 0:
            overall_health = "WARNING (Active critical threats)"
            
        return {
            "timestamp": time.time(),
            "overall_mcps_health": overall_health,
            "key_metrics_snapshot": metrics_overview.get("metrics") if metrics_overview else "Not available",
            "simulated_kubernetes_overview": k8s_sim_status.get("status") if k8s_sim_status else "Not available",
            "active_threats_summary": active_threats_summary if active_threats_summary else "No active threat data"
            # Add more status components as needed
        }

    async def _handle_set_human_oversight_priority(self, args: Dict[str, Any]) -> Dict[str, str]:
        """Handles the 'set_human_oversight_priority' tool call."""
        try:
            # Construct the HumanOversightPriorityRequestModel. Pydantic handles validation.
            priority_request = HumanOversightPriorityRequestModel(**args)
        except Exception as pydantic_error:
            raise ValueError(f"Invalid arguments for set_human_oversight_priority: {pydantic_error}") from pydantic_error
            
        response_dict = await self.oversight_service.set_global_priority(priority_request)
        return response_dict # Should be {"status": "success", "message": "..."}

if __name__ == "__main__":
    # Example Usage (requires mcps_core data_models, services, context_repo, config_loader)
    logging.basicConfig(level=logging.DEBUG)

    # Simplified Mocks for demonstration
    class MockObsService: async def get_observation_for_agent(self, at, aid): return ObservationModel(agent_type=at, agent_id=aid, data={"mock": True})
    class MockActService: async def dispatch_action(self, cmd): return ActionResultModel(command_id=cmd.command_id, status="SUCCESS_SIM", details={"sim_details": cmd.parameters})
    class MockOvrService: async def set_global_priority(self, req): return {"status": "priority_set_sim", "level": req.priority_level}
    class MockCtxRepo(IContextRepository): # Basic implementation
        _data = {"platform_metrics:cluster_summary": {"metrics": {"cpu":0.5}}}
        async def connect(self): pass
        async def disconnect(self): pass
        async def store_context_data(self, k, d, t=None): self._data[k]=d; return True
        async def get_context_data(self, k): return self._data.get(k)
        async def delete_context_data(self, k): self._data.pop(k,None); return True
        async def get_time_series_data(self,m,s,e,a=None): return []
        async def append_to_list(self,k,v,m=None): return 0
        async def get_list_range(self,k,s,e): return []


    async def demo_mcp_tool_handler():
        obs_svc = MockObsService()
        act_svc = MockActService()
        ovr_svc = MockOvrService()
        ctx_repo = MockCtxRepo()

        tool_handler = MCPToolHandler(obs_svc, act_svc, ovr_svc, ctx_repo)

        print("\n--- Listing MCPS Tools ---")
        tools_list = await tool_handler.list_tools()
        print(json.dumps([tool.model_dump(by_alias=True) for tool in tools_list], indent=2))

        print("\n--- Calling 'get_platform_agent_observation' ---")
        obs_args = {"agent_type": "ScaleGuardian", "agent_id": "SG_001"}
        obs_resp = await tool_handler.call_tool("get_platform_agent_observation", obs_args)
        print(json.dumps(obs_resp.model_dump(), indent=2))
        assert obs_resp.status == "SUCCESS"
        assert obs_resp.result["agent_id"] == "SG_001"

        print("\n--- Calling 'execute_platform_agent_action' ---")
        act_args = {
            "agent_id": "DeployMaster_001", "agent_type": "DeployMaster",
            "action_type": "DEPLOY_NEW_VERSION", "parameters": {"service": "voice-agent", "version": "v2.5.1"}
        }
        act_resp = await tool_handler.call_tool("execute_platform_agent_action", act_args)
        print(json.dumps(act_resp.model_dump(), indent=2))
        assert act_resp.status == "SUCCESS"
        assert act_resp.result["details"]["sim_details"]["service"] == "voice-agent"

        print("\n--- Calling 'get_mcps_platform_status' ---")
        stat_resp = await tool_handler.call_tool("get_mcps_platform_status", {})
        print(json.dumps(stat_resp.model_dump(), indent=2))
        assert stat_resp.status == "SUCCESS"
        assert "overall_mcps_health" in stat_resp.result

        print("\n--- Calling 'set_human_oversight_priority' ---")
        priority_args = {
            "priority_level": "HIGH_AVAILABILITY", 
            "details": {"reason": "Upcoming critical event"},
            "target_agent_type": "ScaleGuardian"
        }
        priority_resp = await tool_handler.call_tool("set_human_oversight_priority", priority_args)
        print(json.dumps(priority_resp.model_dump(), indent=2))
        assert priority_resp.status == "SUCCESS"
        assert priority_resp.result["level"] == "HIGH_AVAILABILITY"
        
        print("\n--- Calling a non-existent tool ---")
        bad_tool_resp = await tool_handler.call_tool("non_existent_tool", {})
        print(json.dumps(bad_tool_resp.model_dump(), indent=2))
        assert bad_tool_resp.status == "ERROR"


    # asyncio.run(demo_mcp_tool_handler()) # Comment out if not testing directly
