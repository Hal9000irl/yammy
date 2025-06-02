import asyncio
import os

from mcps.config_loader import MCPSConfigurationService
from mcps.communication_bus import InMemoryMessageBus
from mcps.context_repository import ContextRepository
from mcps.mcps_core.services.context_ingestment import ContextIngestionEngine
from mcps.mcps_core.services.action_dispatch import ActionDispatchService

async def main():
    # Load MCPS configuration
    config_service = MCPSConfigurationService()

    # Initialize message bus (in-memory by default)
    bus = InMemoryMessageBus()
    await bus.connect()

    # Initialize context repository
    context_repo = ContextRepository(config_service, bus)

    # Initialize services
    ingestion_engine = ContextIngestionEngine(config_service, context_repo, bus)
    dispatch_service = ActionDispatchService(config_service, context_repo, bus)

    # Start context ingestion and dispatch pipelines
    await ingestion_engine.start_ingestion_pipelines()
    await dispatch_service.start_dispatching()

    # Keep the service running
    print("MCPS Agent is now running. Press Ctrl+C to exit.")
    stop_event = asyncio.Event()
    try:
        await stop_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        # Clean up
        await ingestion_engine.stop_ingestion_pipelines()
        await bus.disconnect()

if __name__ == '__main__':
    asyncio.run(main()) 