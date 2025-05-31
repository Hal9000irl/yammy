# mcps_core/services/observation.py
# Service to generate observations for MARL agents managing the MCPS platform.

import logging
import time
from typing import Dict, Any

# Absolute imports for mcps package
from mcps.data_models import ObservationModel, MCPSAgentTypeEnum # Using MCPS specific AgentTypeEnum
from mcps.context_repository import IContextRepository
from mcps.config_loader import MCPSConfigurationService

# Configure logger for this module
logger = logging.getLogger("mcps_core.services.observation")

class AgentObservationService:
    """
    Provides tailored observation data to MARL agents that manage the MCPS platform.
    The observations are based on the agent's type/profile and data from the ContextRepository.
    """
    def __init__(self, 
                 config_service: MCPSConfigurationService, 
                 context_repo: IContextRepository):
        self.config_service = config_service
        self.context_repo = context_repo
        logger.info("AgentObservationService initialized.")

    async def get_observation_for_agent(self, 
                                        agent_type: MCPSAgentTypeEnum, 
                                        agent_id: str) -> ObservationModel:
        """
        Generates and returns an observation for a specified MARL agent.
        """
        logger.debug(f"Generating observation for MCPS platform agent_type '{agent_type.value}', id '{agent_id}'.")
        
        # Get the profile for this type of MARL agent from mcps_config.yml
        # This profile defines what data this agent needs to observe.
        agent_profile = self.config_service.get(f"agent_profiles.{agent_type.value}", {})
        observation_keys_config = agent_profile.get("observation_keys", [])
        
        observation_data: Dict[str, Any] = {
            "agent_profile_name": agent_type.value,
            "agent_profile_loaded": bool(agent_profile),
            "current_mcps_timestamp": time.time()
        }
        
        # Fetch configured context keys relevant to this agent's role
        if isinstance(observation_keys_config, list):
            for key_template in observation_keys_config:
                # Allow for dynamic keys if needed, e.g., based on agent_id or target_entity
                # For now, assume keys are direct context keys from the ContextRepository
                context_key = key_template.format(agent_id=agent_id) # Example: "platform_metrics:cluster_summary"
                
                data = await self.context_repo.get_context_data(context_key)
                
                # Use a simplified key in the observation data if desired, or the full key.
                # For instance, "platform_metrics:cluster_summary" might become "platform_metrics_cluster_summary"
                # or just "cluster_summary" if context allows. For simplicity, using a modified key.
                obs_key_name = context_key.replace(":", "_") # Simple replacement
                
                if data is not None:
                    observation_data[obs_key_name] = data
                else:
                    observation_data[obs_key_name] = None # Indicate key was checked but no data found
                    logger.debug(f"No data found in context repository for key '{context_key}' for agent '{agent_id}'.")
        else:
            logger.warning(f"Observation keys for agent type '{agent_type.value}' is not a list in config. Skipping specific key fetching.")

        # Example: Add some generic platform-wide data if no specific keys were configured or found,
        # or if the agent profile dictates it.
        if not observation_keys_config or not any(key.replace(":", "_") in observation_data for key in observation_keys_config):
            logger.debug(f"No specific observation keys found or data retrieved for agent '{agent_id}'. Adding default metrics overview.")
            # This key matches one of the simulated ingestion sources in mcps_config.yml
            latest_platform_metrics = await self.context_repo.get_context_data("platform_metrics:cluster_summary")
            if latest_platform_metrics:
                observation_data["platform_metrics_overview"] = latest_platform_metrics.get("metrics", {})
        
        return ObservationModel(
            agent_type=agent_type,
            agent_id=agent_id,
            data=observation_data,
            # context_version can be used if the structure of 'data' changes over time
        )

if __name__ == "__main__":
    # Example Usage (requires mcps_core.data_models, mcps_core.context_repository, mcps_core.config_loader)
    
    # Setup basic logging for the demo
    logging.basicConfig(level=logging.DEBUG)

    # Mock dependencies for testing
    class MockMCPSConfigService:
        def get(self, key_path, default=None):
            if key_path == "agent_profiles.ScaleGuardian":
                return {"observation_keys": ["platform_metrics:cluster_summary", "kubernetes_api:cluster_status_simulation"]}
            if key_path == "agent_profiles.SecOpsSentinel":
                 return {"observation_keys": ["security_events:recent_simulation"]}
            return default

    class MockContextRepo(IContextRepository):
        _data_store = {
            "platform_metrics:cluster_summary": {"metrics": {"cpu_utilization_percent": 55.0}, "timestamp": time.time()-10},
            "kubernetes_api:cluster_status_simulation": {"status": {"running_pods": 150}, "timestamp": time.time()-5},
            "security_events:recent_simulation": {"events": [{"type": "failed_login", "severity": "HIGH"}]}
        }
        async def connect(self): logger.info("MockContextRepo connected.")
        async def disconnect(self): logger.info("MockContextRepo disconnected.")
        async def store_context_data(self, key, data, ttl_seconds=None): self._data_store[key] = data; return True
        async def get_context_data(self, key): return self._data_store.get(key)
        async def delete_context_data(self, key): return self._data_store.pop(key, None) is not None
        async def get_time_series_data(self, metric_name, start, end, agg=None): return []
        async def append_to_list(self, key, values, max_length=None): return 0
        async def get_list_range(self, key, start, end): return []

    async def demo_observation_service():
        config_service = MockMCPSConfigService()
        context_repo = MockContextRepo()
        await context_repo.connect()

        obs_service = AgentObservationService(config_service, context_repo)

        # Test for ScaleGuardian
        scale_guardian_obs = await obs_service.get_observation_for_agent(
            agent_type=MCPSAgentTypeEnum.SCALE_GUARDIAN,
            agent_id="ScaleGuardian_001"
        )
        print("\n--- Observation for ScaleGuardian_001 ---")
        if scale_guardian_obs:
            print(json.dumps(scale_guardian_obs.model_dump(), indent=2)) # Pydantic v2
        else:
            print("Failed to get observation for ScaleGuardian_001")

        # Test for SecOpsSentinel
        secops_obs = await obs_service.get_observation_for_agent(
            agent_type=MCPSAgentTypeEnum.SECOPS_SENTINEL,
            agent_id="SecOpsSentinel_001"
        )
        print("\n--- Observation for SecOpsSentinel_001 ---")
        if secops_obs:
            print(json.dumps(secops_obs.model_dump(), indent=2))
        else:
            print("Failed to get observation for SecOpsSentinel_001")
        
        # Test for an agent type with no specific keys (should get default overview)
        generic_obs = await obs_service.get_observation_for_agent(
            agent_type=MCPSAgentTypeEnum.GENERIC_MCPS_AGENT, # Assume no profile in MockConfigService
            agent_id="GenericAgent_001"
        )
        print("\n--- Observation for GenericAgent_001 (default) ---")
        if generic_obs:
            print(json.dumps(generic_obs.model_dump(), indent=2))
        else:
            print("Failed to get observation for GenericAgent_001")


        await context_repo.disconnect()

    # To run the demo:
    # asyncio.run(demo_observation_service())
