# mcps_core/communication_bus.py
# Message Bus interface and implementations for the MCPS Platform.

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Any

# Assuming data_models.py is in the same directory or accessible in PYTHONPATH
from .data_models import InterAgentMessageModel # Relative import

# Configure logger for this module
logger = logging.getLogger("mcps_core.communication_bus")

class IMessageBus(ABC):
    """Abstract Base Class for a Message Bus."""
    @abstractmethod
    async def connect(self) -> None:
        """Connects to the message bus infrastructure."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnects from the message bus infrastructure."""
        pass

    @abstractmethod
    async def publish(self, topic: str, message: InterAgentMessageModel) -> None:
        """Publishes a message to a specific topic."""
        pass

    @abstractmethod
    async def subscribe(self, topic: str, callback: Callable[[InterAgentMessageModel], Any]) -> str:
        """
        Subscribes to a topic and registers a callback to be invoked upon message arrival.
        Returns a subscription ID that can be used to unsubscribe.
        """
        pass

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribes from a topic using the subscription ID."""
        pass

    @abstractmethod
    async def send_direct_message(self, recipient_id: str, message: InterAgentMessageModel) -> None:
        """Sends a direct message to a specific recipient (often via a recipient-specific topic)."""
        pass

class InMemoryMessageBus(IMessageBus):
    """
    Simple in-memory message bus implementation for development and testing.
    Not suitable for distributed production environments.
    """
    def __init__(self):
        self._topic_subscribers: Dict[str, Dict[str, Callable[[InterAgentMessageModel], Any]]] = {}
        self._next_sub_id: int = 0
        self._lock = asyncio.Lock() # To protect shared access to subscribers
        logger.info("InMemoryMessageBus initialized.")

    async def connect(self) -> None:
        logger.debug("InMemoryMessageBus connect called (no-op).")
        # No external connection needed for in-memory bus

    async def disconnect(self) -> None:
        logger.debug("InMemoryMessageBus disconnect called (no-op).")
        # No external disconnection needed

    async def publish(self, topic: str, message: InterAgentMessageModel) -> None:
        logger.debug(f"InMemoryBus: Publishing to topic '{topic}': msg_id={message.message_id} from {message.sender_id}")
        async with self._lock:
            # Iterate over a copy of items in case a callback modifies the subscribers (e.g., unsubscribes)
            subscribers_for_topic = list(self._topic_subscribers.get(topic, {}).items())
        
        for sub_id, callback in subscribers_for_topic:
            try:
                # Schedule the callback to run as an independent task
                # This prevents one slow callback from blocking others for the same message.
                asyncio.create_task(self._safe_callback_exec(callback, message, topic, sub_id))
            except Exception as e: # Should ideally not happen if _safe_callback_exec handles it
                logger.error(f"InMemoryBus: Error creating task for callback on topic '{topic}' (sub_id={sub_id}): {e}", exc_info=True)

    async def _safe_callback_exec(self, callback: Callable, message: InterAgentMessageModel, topic: str, sub_id: str):
        try:
            # Check if callback is an async function
            if asyncio.iscoroutinefunction(callback):
                await callback(message)
            else:
                # If it's a synchronous function, run it in a thread pool executor
                # to avoid blocking the asyncio event loop.
                # For simplicity in this blueprint, we'll call it directly,
                # but in production, consider `loop.run_in_executor`.
                callback(message)
        except Exception as e:
            logger.error(f"InMemoryBus: Unhandled error in callback for topic '{topic}' (sub_id={sub_id}): {e}", exc_info=True)


    async def subscribe(self, topic: str, callback: Callable[[InterAgentMessageModel], Any]) -> str:
        async with self._lock:
            if topic not in self._topic_subscribers:
                self._topic_subscribers[topic] = {}
            self._next_sub_id += 1
            subscription_id = f"inmem_sub_{self._next_sub_id}"
            self._topic_subscribers[topic][subscription_id] = callback
        logger.info(f"InMemoryBus: Subscribed to topic '{topic}' with sub_id '{subscription_id}'.")
        return subscription_id

    async def unsubscribe(self, subscription_id: str) -> None:
        async with self._lock:
            found = False
            for topic, subs in self._topic_subscribers.items():
                if subscription_id in subs:
                    del subs[subscription_id]
                    logger.info(f"InMemoryBus: Unsubscribed '{subscription_id}' from topic '{topic}'.")
                    # If the topic has no more subscribers, remove the topic entry
                    if not subs:
                        del self._topic_subscribers[topic]
                    found = True
                    break
            if not found:
                logger.warning(f"InMemoryBus: Subscription ID '{subscription_id}' not found for unsubscription.")
            
    async def send_direct_message(self, recipient_id: str, message: InterAgentMessageModel) -> None:
        # For in-memory, "direct" messages are published to a recipient-specific topic.
        # This allows agents/services to subscribe to their own "inbox" topic.
        direct_topic = f"mcps_direct_{recipient_id}"
        logger.debug(f"InMemoryBus: Sending direct message to '{recipient_id}' via topic '{direct_topic}': msg_id={message.message_id}")
        await self.publish(direct_topic, message)

# Placeholder for RedisPubSubMessageBus
# class RedisPubSubMessageBus(IMessageBus):
#     def __init__(self, redis_url: str, channel_prefix: str = "mcps_pubsub:"):
#         # Needs aioredis library
#         # self.redis_client = aioredis.from_url(redis_url)
#         self.channel_prefix = channel_prefix
#         self.subscribers: Dict[str, asyncio.Task] = {} # subscription_id -> listener_task
#         self.callbacks: Dict[str, Callable] = {} # subscription_id -> callback
#         logger.info(f"RedisPubSubMessageBus initialized for URL: {redis_url} with prefix {channel_prefix}")

#     async def connect(self) -> None:
#         # await self.redis_client.ping() # Ensure connection
#         logger.info("RedisPubSubMessageBus connected (simulated).")

#     async def disconnect(self) -> None:
#         # for task in self.subscribers.values():
#         #     task.cancel()
#         # await asyncio.gather(*self.subscribers.values(), return_exceptions=True)
#         # await self.redis_client.close()
#         logger.info("RedisPubSubMessageBus disconnected (simulated).")

#     async def publish(self, topic: str, message: InterAgentMessageModel) -> None:
#         # channel = self.channel_prefix + topic
#         # await self.redis_client.publish(channel, message.model_dump_json())
#         logger.debug(f"RedisPubSub: Published to channel for topic '{topic}' (simulated).")

#     async def subscribe(self, topic: str, callback: Callable[[InterAgentMessageModel], Any]) -> str:
#         # subscription_id = f"redis_sub_{topic}_{uuid.uuid4().hex[:6]}"
#         # channel = self.channel_prefix + topic
#         # self.callbacks[subscription_id] = callback
#         # pubsub = self.redis_client.pubsub()
#         # await pubsub.subscribe(channel)
#         # self.subscribers[subscription_id] = asyncio.create_task(self._listener(pubsub, subscription_id))
#         logger.info(f"RedisPubSub: Subscribed to topic '{topic}' (simulated).")
#         return "simulated_redis_sub_id"

#     async def _listener(self, pubsub, subscription_id: str):
#         # try:
#         #     async for message_data in pubsub.listen():
#         #         if message_data["type"] == "message":
#         #             payload = json.loads(message_data["data"])
#         #             iam_message = InterAgentMessageModel(**payload)
#         #             if subscription_id in self.callbacks:
#         #                 asyncio.create_task(self.callbacks[subscription_id](iam_message))
#         # except asyncio.CancelledError:
#         #     logger.info(f"Redis listener for {subscription_id} cancelled.")
#         # except Exception as e:
#         #     logger.error(f"Redis listener error for {subscription_id}: {e}", exc_info=True)
#         # finally:
#         #     await pubsub.unsubscribe()
#         #     await pubsub.close()
#         pass

#     async def unsubscribe(self, subscription_id: str) -> None:
#         # if subscription_id in self.subscribers:
#         #     self.subscribers[subscription_id].cancel()
#         #     del self.subscribers[subscription_id]
#         #     del self.callbacks[subscription_id]
#         logger.info(f"RedisPubSub: Unsubscribed '{subscription_id}' (simulated).")

#     async def send_direct_message(self, recipient_id: str, message: InterAgentMessageModel) -> None:
#         direct_topic = f"direct_{recipient_id}"
#         await self.publish(direct_topic, message)


# Placeholder for KafkaMessageBus
# class KafkaMessageBus(IMessageBus):
#     def __init__(self, bootstrap_servers: str, default_topic_prefix: str = "mcps_events"):
#         # Needs aiokafka library
#         # self.producer = AIOKafkaProducer(bootstrap_servers=bootstrap_servers)
#         # self.consumers: Dict[str, AIOKafkaConsumer] = {} # topic -> consumer
#         # self.callbacks: Dict[str, List[Callable]] = {} # topic -> list of callbacks
#         logger.info(f"KafkaMessageBus initialized for servers: {bootstrap_servers}")

#     async def connect(self) -> None: # await self.producer.start()
#     async def disconnect(self) -> None: # await self.producer.stop(); for c in self.consumers.values(): await c.stop()
#     async def publish(self, topic: str, message: InterAgentMessageModel) -> None: # await self.producer.send_and_wait(topic, message.model_dump_json().encode())
#     async def subscribe(self, topic: str, callback: Callable[[InterAgentMessageModel], Any]) -> str: # Create/get consumer, start listener task
#     async def unsubscribe(self, subscription_id: str) -> None: # More complex, might involve stopping consumer or removing callback
#     async def send_direct_message(self, recipient_id: str, message: InterAgentMessageModel) -> None: # Publish to a specific partition or topic
#         pass

if __name__ == "__main__":
    # Example Usage (requires data_models.py to be accessible)
    async def sample_callback(message: InterAgentMessageModel):
        print(f"Callback received on topic '{message.topic}': {message.payload} from {message.sender_id}")

    async def direct_message_handler(message: InterAgentMessageModel):
        print(f"Direct message received by Agent1 from '{message.sender_id}': {message.payload}")

    async def demo():
        bus = InMemoryMessageBus()
        await bus.connect()

        sub_id1 = await bus.subscribe("general_updates", sample_callback)
        sub_id2 = await bus.subscribe("general_updates", lambda m: print(f"Lambda Callback: {m.payload}"))
        
        # Agent1 subscribes to its direct messages
        agent1_direct_sub_id = await bus.subscribe("mcps_direct_Agent1", direct_message_handler)

        msg1 = InterAgentMessageModel(sender_id="ServiceA", topic="general_updates", payload={"data": "Update 1"})
        msg2 = InterAgentMessageModel(sender_id="ServiceB", topic="general_updates", payload={"data": "Update 2"})
        direct_msg_to_agent1 = InterAgentMessageModel(sender_id="ServiceC", recipient_id="Agent1", payload={"task": "process_this"})

        await bus.publish("general_updates", msg1)
        await bus.publish("another_topic", msg2) # No subscriber, should be silent
        await bus.send_direct_message("Agent1", direct_msg_to_agent1)
        await bus.send_direct_message("Agent2", InterAgentMessageModel(sender_id="ServiceD", recipient_id="Agent2", payload={"info": "for agent 2"}))


        await asyncio.sleep(0.1) # Allow callbacks to process

        await bus.unsubscribe(sub_id1)
        print("Unsubscribed sub_id1")
        await bus.publish("general_updates", InterAgentMessageModel(sender_id="ServiceC", topic="general_updates", payload={"data": "Update 3 after unsub"}))
        
        await asyncio.sleep(0.1)
        await bus.disconnect()

    asyncio.run(demo())
