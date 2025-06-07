# mcps_core/services/context_ingestion.py
# Context Ingestion Engine service for the MCPS Platform.

import asyncio
import logging
import random
import time
import uuid # For generating unique IDs for simulated events
from typing import Dict, Any, Optional, Callable, List

# Relative imports for modules within the mcps package
from .data_models import InterAgentMessageModel
from .communication_bus import IMessageBus
from .config_loader import MCPSConfigurationService
from typing import Any # Added for IContextRepository placeholder
IContextRepository = Any # Placeholder to resolve NameError

# IContextRepository is likely defined in this file or an ABC, an import like this is not needed.
# from . import IContextRepository # If IContextRepository were in mcps/__init__.py

# Configure logger for this module
logger = logging.getLogger("mcps_core.services.context_ingestion")

class ContextIngestionEngine:
    """
    Ingests context data from various simulated or real platform sources
    into the ContextRepository and publishes relevant events to the MessageBus.
    """
    def __init__(self,
                 config_service: MCPSConfigurationService,
                 context_repo: IContextRepository,
                 message_bus: IMessageBus):
        self.config_service = config_service
        self.context_repo = context_repo
        self.message_bus = message_bus
        self._running_tasks: List[asyncio.Task] = []
        
        # Handlers for different types of ingestion sources defined in mcps_config.yml
        self._ingestion_source_handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {
            "platform_metrics_simulation": self._simulate_platform_metrics,
            "kubernetes_data_simulation": self._simulate_k8s_data, # Renamed from mcp_Marl_aiempaths(1).py
            "security_events_simulation": self._simulate_security_events,
            "file_poller": self._poll_file_source, # Example for security audit log ingestion
            # "prometheus_scraper": self._scrape_prometheus_source, # Placeholder for real scraper
            # "kubernetes_api": self._fetch_kubernetes_api_data, # Placeholder for real K8s API
        }
        logger.info("ContextIngestionEngine initialized.")

    async def _simulate_platform_metrics(self, source_config: Dict[str, Any]):
        """Simulates ingesting platform metrics (CPU, memory, network, etc.)."""
        source_name = source_config.get('name', 'unnamed_metrics_sim')
        logger.debug(f"Simulating platform metrics ingestion for source: {source_name}")
        
        metrics = {
            "cpu_utilization_percent": round(random.uniform(10.0, 90.0), 2),
            "memory_usage_gb": round(random.uniform(
                self.config_service.get(f"agent_profiles.ScaleGuardian.min_memory_gb", 1.0), # Example dynamic config
                self.config_service.get(f"agent_profiles.ScaleGuardian.max_memory_gb", 16.0)
            ), 2),
            "active_connections": random.randint(50, 500),
            "api_latency_ms_p95": round(random.uniform(50, 300), 1),
            "disk_io_read_mbps": round(random.uniform(10, 100), 1),
            "disk_io_write_mbps": round(random.uniform(5, 50), 1),
            "network_throughput_gbps": round(random.uniform(0.1, 1.0), 2)
        }
        context_key = source_config.get("context_key", "platform_metrics:cluster_summary") # From mcps_config.yml
        
        await self.context_repo.store_context_data(
            context_key,
            {"metrics": metrics, "timestamp": time.time(), "source": source_name},
            ttl_seconds=source_config.get("ttl_sec", self.config_service.get("context_repository.default_ttl_seconds", 300))
        )
        
        await self.message_bus.publish(
            topic=source_config.get("publish_topic", "mcps_platform_context_updates"),
            message=InterAgentMessageModel(
                sender_id="ContextIngestionEngine",
                topic=source_config.get("publish_topic", "mcps_platform_context_updates"), # Ensure topic is in message
                payload={"type": "platform_metrics_update", "source": source_name, "data": metrics}
            )
        )

    async def _simulate_k8s_data(self, source_config: Dict[str, Any]):
        """Simulates ingesting Kubernetes cluster data."""
        source_name = source_config.get('name', 'unnamed_k8s_sim')
        logger.debug(f"Simulating Kubernetes data ingestion for source: {source_name}")
        
        total_nodes = random.randint(3, 10)
        ready_nodes = random.randint(max(1, total_nodes - 2), total_nodes) # Ensure ready <= total & at least 1
        
        k8s_data = {
            "total_nodes": total_nodes,
            "ready_nodes": ready_nodes,
            "unready_nodes": total_nodes - ready_nodes,
            "running_pods": random.randint(20, 200),
            "pending_pods": random.randint(0, 10),
            "failed_pods": random.randint(0, 5),
            "cluster_cpu_capacity_cores": float(total_nodes * self.config_service.get("simulations.k8s.cores_per_node", 8.0)),
            "cluster_cpu_usage_cores": round(random.uniform(total_nodes * 1.0, total_nodes * 6.0), 1), # Simulate usage
            "cluster_memory_capacity_gb": float(total_nodes * self.config_service.get("simulations.k8s.memory_gb_per_node", 32.0)),
            "cluster_memory_usage_gb": round(random.uniform(total_nodes * 4.0, total_nodes * 28.0), 1),
        }
        k8s_data["cluster_cpu_usage_percent"] = round((k8s_data["cluster_cpu_usage_cores"] / k8s_data["cluster_cpu_capacity_cores"]) * 100, 1) if k8s_data["cluster_cpu_capacity_cores"] > 0 else 0
        k8s_data["cluster_memory_usage_percent"] = round((k8s_data["cluster_memory_usage_gb"] / k8s_data["cluster_memory_capacity_gb"]) * 100, 1) if k8s_data["cluster_memory_capacity_gb"] > 0 else 0

        context_key = source_config.get("context_key", "kubernetes_api:cluster_status_simulation")
        await self.context_repo.store_context_data(
            context_key,
            {"status": k8s_data, "timestamp": time.time(), "source": source_name},
            ttl_seconds=source_config.get("ttl_sec", 300)
        )
        
        await self.message_bus.publish(
            topic=source_config.get("publish_topic", "mcps_platform_context_updates"),
            message=InterAgentMessageModel(
                sender_id="ContextIngestionEngine",
                topic=source_config.get("publish_topic", "mcps_platform_context_updates"),
                payload={"type": "kubernetes_status_update", "source": source_name, "data": k8s_data}
            )
        )

    async def _simulate_security_events(self, source_config: Dict[str, Any]):
        """Simulates ingesting security-related events."""
        source_name = source_config.get('name', 'unnamed_security_sim')
        logger.debug(f"Simulating security events ingestion for source: {source_name}")
        events = []
        if random.random() < source_config.get("event_probability", 0.2): # Chance of an event
            event_count = random.randint(1, source_config.get("max_events_per_interval", 3))
            for _ in range(event_count):
                events.append({
                    "event_id": str(uuid.uuid4()),
                    "type": random.choice(["failed_login_attempt", "suspicious_api_call", "firewall_block", "malware_detected", "data_exfiltration_attempt"]),
                    "severity": random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
                    "timestamp": time.time() - random.uniform(0, source_config.get("interval_sec", 60)), # Event time within interval
                    "source_ip": f"10.0.{random.randint(1,254)}.{random.randint(1,254)}", # Example internal IP
                    "target_resource": f"service_{random.choice(['auth', 'payment', 'user_data', 'voice_agent_core'])}",
                    "details": {"user_agent": "SimulatedAttacker/1.0", "path": f"/api/v1/resource{random.randint(1,10)}"}
                })
        
        context_key = source_config.get("context_key", "security_events:recent_simulation")
        # Append to a list in Redis, potentially capped by max_length
        # The context_repo.append_to_list should handle JSON serialization of individual event dicts
        if events: # Only append if there are new events
            await self.context_repo.append_to_list(context_key, events, max_length=source_config.get("max_list_length", 50))

            await self.message_bus.publish(
                topic=source_config.get("publish_topic", "mcps_security_alerts"),
                message=InterAgentMessageModel(
                    sender_id="ContextIngestionEngine",
                    topic=source_config.get("publish_topic", "mcps_security_alerts"),
                    payload={"type": "new_security_events", "source": source_name, "events": events}
                )
            )

    async def _poll_file_source(self, source_config: Dict[str, Any]):
        """Example handler to poll a directory for new files (e.g., audit logs)."""
        source_name = source_config.get('name', 'unnamed_file_poller')
        file_path = source_config.get('path') # Directory to poll
        context_key = source_config.get('context_key')
        
        if not file_path or not context_key:
            logger.error(f"File poller source '{source_name}' is missing 'path' or 'context_key' in config.")
            return
            
        logger.debug(f"Polling file source '{source_name}' at path: {file_path}")
        # This is a simplified simulation. Real implementation would:
        # - List files in the directory.
        # - Keep track of processed files (e.g., by renaming, moving, or storing hashes).
        # - Read new files, parse their content.
        # - Store relevant data in context_repo and publish messages.
        # For now, let's simulate finding one new "audit report".
        if random.random() < 0.1: # 10% chance of finding a new "report"
            simulated_report_content = {
                "report_id": f"audit_{uuid.uuid4().hex[:8]}",
                "timestamp": time.time(),
                "summary": "Simulated security audit completed.",
                "findings_critical": random.randint(0,1),
                "findings_high": random.randint(0,3)
            }
            await self.context_repo.store_context_data(
                context_key, # e.g., "security_audits:latest_report"
                simulated_report_content,
                ttl_seconds=source_config.get("ttl_sec", 86400) # Keep for a day
            )
            await self.message_bus.publish(
                topic=source_config.get("publish_topic", "mcps_platform_context_updates"),
                message=InterAgentMessageModel(
                    sender_id="ContextIngestionEngine",
                    topic=source_config.get("publish_topic", "mcps_platform_context_updates"),
                    payload={"type": "new_audit_report", "source": source_name, "report": simulated_report_content}
                )
            )
            logger.info(f"FilePoller '{source_name}': Ingested simulated report {simulated_report_content['report_id']}")


    async def _periodic_ingestion_task_runner(self, source_name: str, source_config: Dict[str, Any]):
        """Generic task runner for a single ingestion source."""
        source_type = source_config.get("type")
        interval_seconds = source_config.get("interval_sec", 60) # Default to 60 seconds

        handler = self._ingestion_source_handlers.get(source_type)
        if not handler:
            logger.error(f"No handler found for ingestion source type: '{source_type}' (source: '{source_name}')")
            return

        logger.info(f"Starting periodic ingestion for source '{source_name}' (type: '{source_type}'), interval: {interval_seconds}s")
        while True:
            try:
                logger.debug(f"Running ingestion handler for source '{source_name}'")
                await handler(source_config) # Pass the full source_config to the handler
            except asyncio.CancelledError:
                logger.info(f"Ingestion task for source '{source_name}' cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in periodic ingestion task for source '{source_name}': {e}", exc_info=True)
            
            await asyncio.sleep(interval_seconds)

    async def start_ingestion_pipelines(self) -> None:
        """Starts all enabled ingestion pipelines based on the configuration."""
        # The configuration for sources is now expected to be a dictionary
        # where keys are source names and values are their configurations.
        ingestion_sources_map = self.config_service.get("context_ingestion_sources", {})
        if not isinstance(ingestion_sources_map, dict):
            logger.error("Invalid 'context_ingestion_sources' configuration: expected a dictionary (map).")
            return

        logger.info(f"Starting context ingestion pipelines for {len(ingestion_sources_map)} configured sources...")
        for source_name, source_config in ingestion_sources_map.items():
            if not isinstance(source_config, dict):
                logger.warning(f"Skipping invalid source configuration for '{source_name}': not a dictionary.")
                continue
            
            # Add the source name to its own config dict if not already present, for handler convenience
            source_config.setdefault("name", source_name)

            if source_config.get("enabled", False):
                task = asyncio.create_task(self._periodic_ingestion_task_runner(source_name, source_config))
                self._running_tasks.append(task)
            else:
                logger.info(f"Ingestion source '{source_name}' (type: {source_config.get('type')}) is disabled.")
        
        if not self._running_tasks:
            logger.warning("No enabled ingestion sources found or configured in context_ingestion_sources.")

    async def stop_ingestion_pipelines(self) -> None:
        """Stops all running ingestion pipeline tasks."""
        logger.info(f"Stopping {len(self._running_tasks)} context ingestion pipeline tasks...")
        for task in self._running_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for all tasks to complete or be cancelled
        results = await asyncio.gather(*self._running_tasks, return_exceptions=True)
        for i, result in enumerate(results):
            # Log any exceptions that weren't CancelledError
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                # Attempt to find the source name for better logging, though task objects don't easily store it
                logger.error(f"Exception during ingestion task shutdown (task index {i}): {result}", exc_info=result)
        
        self._running_tasks.clear()
        logger.info("All context ingestion pipelines stopped.")

if __name__ == "__main__":
    # Example Usage (requires data_models, communication_bus, context_repository, config_loader)
    # This is a more complex setup for direct testing.
    # You'd need to mock or provide real instances of dependencies.

    async def demo_ingestion():
        logging.basicConfig(level=logging.DEBUG)

        # Dummy config for testing
        class MockConfigService:
            def get(self, key_path, default=None):
                if key_path == "context_ingestion_sources":
                    return {
                        "sim_metrics": {"enabled": True, "type": "platform_metrics_simulation", "interval_sec": 3, "context_key": "test_metrics"},
                        "sim_k8s": {"enabled": True, "type": "kubernetes_data_simulation", "interval_sec": 4, "context_key": "test_k8s"},
                        "sim_security": {"enabled": True, "type": "security_events_simulation", "interval_sec": 2, "context_key": "test_security_events"}
                    }
                if key_path == "context_repository.default_ttl_seconds":
                    return 60
                return default
        
        class MockContextRepo(IContextRepository): # Implement methods or use a real one
            async def connect(self): print("MockRepo: Connect")
            async def disconnect(self): print("MockRepo: Disconnect")
            async def store_context_data(self, key, data, ttl_seconds=None): print(f"MockRepo: Store {key} = {str(data)[:50]}... (TTL: {ttl_seconds})")
            async def get_context_data(self, key): return None
            async def delete_context_data(self, key): return True
            async def get_time_series_data(self, metric_name, start, end, agg=None): return []
            async def append_to_list(self, key, values, max_length=None): print(f"MockRepo: Append to {key}: {len(values)} items"); return len(values)
            async def get_list_range(self, key, start, end): return []


        mock_config = MockConfigService()
        mock_bus = InMemoryMessageBus() # From communication_bus.py
        await mock_bus.connect()
        mock_repo = MockContextRepo() # Use your RedisContextRepository for real test
        await mock_repo.connect()

        engine = ContextIngestionEngine(mock_config, mock_repo, mock_bus)
        await engine.start_ingestion_pipelines()

        print("Ingestion pipelines started. Running for 10 seconds...")
        await asyncio.sleep(10)

        await engine.stop_ingestion_pipelines()
        await mock_bus.disconnect()
        await mock_repo.disconnect()
        print("Ingestion demo finished.")

    # asyncio.run(demo_ingestion()) # Comment out if not testing directly
