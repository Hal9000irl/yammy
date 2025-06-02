# mcps_core/services/configuration.py
# API service for managing the MCPS platform's configuration.

import logging
from typing import Dict, Any, Optional

# Absolute imports for mcps package
from mcps.config_loader import MCPSConfigurationService # To interact with the loaded config
from mcps.data_models import ConfigSectionUpdateRequestModel # For request body validation (used by API layer)

# Configure logger for this module
logger = logging.getLogger("mcps_core.services.configuration")

class ConfigurationApiService:
    """
    Provides methods to interact with the MCPS platform's configuration.
    This service is typically exposed via API endpoints in mcps_core/main.py.
    """
    def __init__(self, config_service: MCPSConfigurationService):
        self.config_service = config_service
        logger.info("ConfigurationApiService initialized.")

    async def get_full_mcps_configuration(self) -> Dict[str, Any]:
        """
        Retrieves the entire current configuration of the MCPS platform.
        Sensitive values should be redacted if this endpoint is not strictly admin-only.
        """
        logger.debug("Fetching full MCPS configuration.")
        # Consider redacting sensitive keys (e.g., jwt_secret_key, database passwords)
        # before returning if this endpoint is not highly secured.
        # For simplicity, returning all for now.
        return self.config_service.get_all_config()

    async def get_mcps_config_section(self, section_key_path: str) -> Optional[Any]:
        """
        Retrieves a specific section or key from the MCPS configuration
        using a dot-separated path.
        Returns None if the path is not found.
        """
        logger.debug(f"Fetching MCPS configuration for section/key: '{section_key_path}'.")
        value = self.config_service.get(section_key_path)
        if value is None:
            # Check if the key path actually exists vs. value being None
            # This check can be complex if we want to distinguish truly missing path vs. a key with null value.
            # For now, if get returns None, we assume it's not found or explicitly null.
            logger.warning(f"Configuration section/key '{section_key_path}' not found or is null.")
        return value

    async def update_mcps_config_section(self, section_key_path: str, new_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Updates a specific section of the MCPS configuration.
        The 'new_data' will replace or update the contents of the specified section.
        """
        logger.info(f"Request to update MCPS configuration section: '{section_key_path}'.")
        
        # The MCPSConfigurationService.update_section_config handles the actual update and save.
        # It expects the full data for the section to be updated/replaced.
        success = await self.config_service.update_section_config(section_key_path, new_data)
        
        if success:
            # Potentially trigger a notification on the message bus that config has changed,
            # so other services can reload if they support dynamic config updates.
            # Example:
            # await self.message_bus.publish("mcps_config_updated_events", 
            #    InterAgentMessageModel(sender_id="ConfigurationApiService", 
            #                           topic="mcps_config_updated_events", 
            #                           payload={"updated_section": section_key_path}))
            logger.info(f"MCPS configuration section '{section_key_path}' updated successfully.")
            return {"status": "success", "message": f"Configuration section '{section_key_path}' updated."}
        else:
            logger.error(f"Failed to update MCPS configuration section '{section_key_path}'. Check logs for details.")
            # The config_loader already logs errors during save.
            return {"status": "error", "message": f"Failed to update configuration section '{section_key_path}'."}

if __name__ == "__main__":
    # Example Usage (requires mcps_core.config_loader and dummy mcps_config.yml)
    import asyncio
    import os
    import yaml

    logging.basicConfig(level=logging.DEBUG)

    # Create a dummy mcps_config.yml for testing
    dummy_config_path = "temp_mcps_config_for_test.yml"
    initial_config_data = {
        "api_server": {"port": 8080, "log_level": "DEBUG"},
        "feature_flags": {"new_dashboard": True, "experimental_scaling": False}
    }
    with open(dummy_config_path, "w") as f:
        yaml.dump(initial_config_data, f)

    async def demo_config_service():
        config_service_instance = MCPSConfigurationService(config_file_path=dummy_config_path)
        api_config_service = ConfigurationApiService(config_service_instance)

        print("\n--- Testing Get Full Config ---")
        full_conf = await api_config_service.get_full_mcps_configuration()
        print(json.dumps(full_conf, indent=2))
        assert full_conf["api_server"]["port"] == 8080

        print("\n--- Testing Get Config Section (api_server) ---")
        api_server_conf = await api_config_service.get_mcps_config_section("api_server")
        print(json.dumps(api_server_conf, indent=2))
        assert api_server_conf["port"] == 8080

        print("\n--- Testing Get Config Key (api_server.port) ---")
        port_val = await api_config_service.get_mcps_config_section("api_server.port")
        print(f"Port: {port_val}")
        assert port_val == 8080
        
        print("\n--- Testing Get Non-existent Section ---")
        non_existent = await api_config_service.get_mcps_config_section("non_existent_section")
        print(f"Non-existent section: {non_existent}")
        assert non_existent is None

        print("\n--- Testing Update Config Section (feature_flags) ---")
        update_data = {"experimental_scaling": True, "beta_feature_X": "enabled"}
        update_result = await api_config_service.update_mcps_config_section("feature_flags", update_data)
        print(f"Update Result: {update_result}")
        assert update_result["status"] == "success"

        updated_feature_flags = await api_config_service.get_mcps_config_section("feature_flags")
        print("Updated feature_flags:")
        print(json.dumps(updated_feature_flags, indent=2))
        assert updated_feature_flags["experimental_scaling"] is True
        assert updated_feature_flags["new_dashboard"] is True # Original value should persist if not overwritten
        assert updated_feature_flags["beta_feature_X"] == "enabled"

        print("\n--- Testing Update Nested Config Key (api_server.log_level) ---")
        # Note: update_mcps_config_section expects a dictionary for new_data.
        # To update a single key in a nested dict, you'd pass the whole sub-dict or modify update_section_config.
        # The current MCPSConfigurationService.update_section_config will replace/merge at the specified section_key_path.
        # If section_key_path is "api_server.log_level", it might create a sub-dict if not handled carefully.
        # Let's assume we want to update the 'api_server' section with a new log_level.
        update_api_server_data = {"log_level": "WARNING", "timeout_sec": 60} # This will merge into api_server
        update_result_api = await api_config_service.update_mcps_config_section("api_server", update_api_server_data)
        print(f"Update API Server Result: {update_result_api}")
        updated_api_server = await api_config_service.get_mcps_config_section("api_server")
        print("Updated api_server:")
        print(json.dumps(updated_api_server, indent=2))
        assert updated_api_server["log_level"] == "WARNING"
        assert updated_api_server["port"] == 8080 # Original port should persist
        assert updated_api_server["timeout_sec"] == 60


        # Clean up dummy config
        if os.path.exists(dummy_config_path):
            os.remove(dummy_config_path)

    # asyncio.run(demo_config_service()) # Comment out if not testing directly
