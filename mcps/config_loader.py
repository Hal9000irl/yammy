# mcps_core/config_loader.py
# Configuration Management for the MCPS Platform.

import os
import yaml
import logging
from typing import Dict, Any, Optional

config_logger = logging.getLogger("mcps_core.config_loader") # Use a distinct logger

DEFAULT_MCPS_CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "mcps_config.yml")

class MCPSConfigurationService:
    """Manages MCPS platform configuration from a YAML file."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MCPSConfigurationService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_file_path: str = DEFAULT_MCPS_CONFIG_FILE_PATH):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self.config_file_path = os.path.abspath(config_file_path)
        self.config: Dict[str, Any] = self._load_config()
        config_logger.info(f"MCPSConfigurationService initialized. Loaded config from: {self.config_file_path}")
        self._initialized = True

    def _default_config(self) -> Dict[str, Any]:
        """Provides a basic default configuration structure for MCPS."""
        return {
            "api_server": {"host": "0.0.0.0", "port": 8000, "log_level": "INFO"},
            "redis": {"url": "redis://localhost:6379/0", "max_connections": 50},
            "message_bus": {"type": "in_memory"},
            "context_repository": {"default_ttl_seconds": 3600},
            "agent_profiles": {},
            "context_ingestion_sources": [],
            "observability": {"metrics_endpoint_enabled": True},
            "performance_tuning": {
                "connection_pooling": {"http_max_connections": 100, "http_timeout_seconds": 30},
                "caching": {"memory_max_size_items": 1000, "memory_default_ttl_seconds": 300},
                "batch_processing": {"default_batch_size": 100, "default_flush_interval_seconds": 5.0}
            },
            "security_framework": {
                "jwt_secret_key": "pleas_change_this_secret_key_in_production",
                "jwt_algorithm": "HS256",
                "jwt_token_expire_minutes": 30
            },
            "tenancy": {"enabled": False, "default_tier": "starter"}
        }

    def _load_config(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.config_file_path):
                with open(self.config_file_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    if loaded_config:
                        config_logger.debug(f"Successfully loaded MCPS configuration from {self.config_file_path}")
                        # Deep merge with defaults to ensure all keys exist might be good here
                        # For simplicity, we'll assume the loaded config is comprehensive or keys are checked at point of use.
                        return loaded_config
                    config_logger.warning(f"MCPS Config file {self.config_file_path} is empty. Using defaults.")
                    return self._default_config()
            else:
                config_logger.warning(f"MCPS Config file not found at {self.config_file_path}. Using defaults and creating one.")
                default_conf = self._default_config()
                self._save_config(default_conf) # Save a default one
                return default_conf
        except Exception as e:
            config_logger.error(f"Error loading MCPS config from {self.config_file_path}: {e}. Using defaults.", exc_info=True)
            return self._default_config()

    def _save_config(self, config_data: Dict[str, Any]) -> None:
        """Saves the current configuration to the YAML file."""
        try:
            os.makedirs(os.path.dirname(self.config_file_path), exist_ok=True)
            with open(self.config_file_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            config_logger.info(f"MCPS Configuration saved to {self.config_file_path}")
        except Exception as e:
            config_logger.error(f"Error saving MCPS config to {self.config_file_path}: {e}", exc_info=True)
    
    def get(self, key_path: str, default: Optional[Any] = None) -> Any:
        """
        Retrieves a configuration value using a dot-separated key_path.
        Example: get('api_server.port', 8080)
        """
        keys = key_path.split('.')
        val = self.config
        try:
            for key in keys:
                if isinstance(val, dict):
                    val = val[key]
                else: # Key path leads to a non-dict intermediate value
                    config_logger.debug(f"Key path '{key_path}' not fully found (intermediate not dict). Returning default.")
                    return default
            return val
        except KeyError:
            config_logger.debug(f"Key '{key}' not found in path '{key_path}'. Returning default.")
            return default
        except Exception as e:
            config_logger.error(f"Error accessing key_path '{key_path}': {e}. Returning default.")
            return default


    def get_all_config(self) -> Dict[str, Any]:
        """Returns a copy of the entire configuration."""
        return self.config.copy()

    async def update_section_config(self, section_key_path: str, data: Dict[str, Any]) -> bool:
        """
        Updates a section of the configuration and saves it.
        section_key_path can be dot-separated for nested sections.
        """
        config_logger.info(f"Attempting to update MCPS config section: {section_key_path}")
        
        keys = section_key_path.split('.')
        current_level = self.config
        
        try:
            for i, key in enumerate(keys[:-1]): # Navigate to the parent of the target section
                if key not in current_level or not isinstance(current_level[key], dict):
                    # If path doesn't exist or is not a dict, create it
                    current_level[key] = {}
                current_level = current_level[key]
            
            target_key = keys[-1]
            if target_key not in current_level or not isinstance(current_level[target_key], dict):
                # If target section doesn't exist or isn't a dict, replace it
                current_level[target_key] = data
            else:
                # If it exists and is a dict, update it (shallow merge)
                current_level[target_key].update(data)

            self._save_config(self.config)
            # TODO: In a real system, publish a "config_updated" event on the message bus
            # for dynamic reloading by interested services.
            # Example: await self.message_bus.publish("mcps_config_updates", 
            # InterAgentMessageModel(sender_id="ConfigurationService", 
            # topic="mcps_config_updates", payload={"updated_section": section_key_path}))
            config_logger.info(f"MCPS Config section '{section_key_path}' updated and saved.")
            return True
        except Exception as e:
            config_logger.error(f"Failed to update or save config section {section_key_path}: {e}", exc_info=True)
            # Optionally, reload original config to revert changes if save failed
            # self.config = self._load_config() 
            return False

if __name__ == '__main__':
    # Example Usage:
    # Ensure mcps_config.yml exists or a default one will be created.
    config_service = MCPSConfigurationService()
    print("Current API Port:", config_service.get("api_server.port", "N/A"))
    print("Redis URL:", config_service.get("redis.url"))
    print("Non-existent key:", config_service.get("some.made.up.key", "DefaultValueForMissing"))

    # Example update (this is synchronous for demonstration, but API endpoint would be async)
    # import asyncio
    # async def demo_update():
    #     success = await config_service.update_section_config("api_server", {"port": 8001, "new_setting": True})
    #     print("Update successful:", success)
    #     print("New API Port:", config_service.get("api_server.port"))
    # asyncio.run(demo_update())
