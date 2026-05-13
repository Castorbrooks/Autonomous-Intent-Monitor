#!/usr/bin/env python3
"""
Autonomous Intent Monitor
-------------------------
Institutional-grade asynchronous system that scans social media sources (X/Twitter, Reddit),
classifies post intents using an LLM (Gemini or DeepSeek), and demonstrates a polyglot bridge
to high-performance microservices (Rust/Node.js). Designed for auditability, extensibility,
and production deployment.

Environment variables:
    LLM_PROVIDER          : 'gemini' or 'deepseek' (default: 'gemini')
    LLM_API_KEY           : API key for the chosen LLM provider
    MICROSERVICE_URL      : Optional endpoint for polyglot bridge (default: http://localhost:8080/intents)
    LOG_LEVEL             : Logging level (DEBUG, INFO, WARNING, ERROR, default: INFO)
    SIMULATION_RATE_LIMIT : Seconds between post generations per source (default: 5.0)

Usage:
    python autonomous_intent_monitor.py
"""

import asyncio
import json
import logging
import os
import random
import signal
import sys
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, AsyncGenerator, Union

import httpx
from pydantic import BaseModel, Field, HttpUrl, ValidationError
from dotenv import load_dotenv

load_dotenv()


# -----------------------------------------------------------------------------
# Pydantic Models for Strict Data Validation
# -----------------------------------------------------------------------------
class SourceType(str, Enum):
    TWITTER = "twitter"
    REDDIT = "reddit"

class Post(BaseModel):
    """Raw post from a social media source."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: SourceType
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

class IntentLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class ClassifiedIntent(BaseModel):
    """Post enriched with LLM classification."""
    post: Post
    intent_level: IntentLevel
    confidence_score: float = Field(ge=0.0, le=1.0)
    llm_response_raw: str
    classification_timestamp: datetime = Field(default_factory=datetime.utcnow)

class BridgePayload(BaseModel):
    """Payload format for polyglot microservice bridge."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    classified_intents: List[ClassifiedIntent]
    source_system: str = "autonomous-intent-monitor-python"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# -----------------------------------------------------------------------------
# Logging Configuration (Institutional Audit Trail)
# -----------------------------------------------------------------------------
class AuditLogger:
    _instance = None
    _logger = None

    @classmethod
    def get_logger(cls) -> logging.Logger:
        if cls._instance is None:
            cls._instance = cls._setup_logger()
        return cls._instance

    @staticmethod
    def _setup_logger() -> logging.Logger:
        logger = logging.getLogger("IntentMonitor")
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, log_level, logging.INFO))

        # Prevent duplicate handlers
        if logger.handlers:
            return logger

        # Console handler with structured format
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Rotating file handler for audit persistence
        try:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                "intent_monitor_audit.log",
                maxBytes=10_485_760,  # 10 MB
                backupCount=10
            )
            file_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(module)s | %(funcName)s | %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.error(f"Failed to create file audit handler: {e}")

        return logger


# -----------------------------------------------------------------------------
# AI Integration: LLM Intent Classifier (Gemini / DeepSeek)
# -----------------------------------------------------------------------------
class IntentClassifier:
    """Asynchronous classifier that sends posts to an LLM API and extracts intent level."""

    SYSTEM_PROMPT = (
        "You are an intent classification engine for financial and security monitoring. "
        "Classify the following social media post into one of three intent levels: "
        "HIGH (urgent threat, crisis, immediate action required), "
        "MEDIUM (notable trend, requires attention but not urgent), "
        "LOW (routine discussion, informational, no action needed). "
        "Respond only with a JSON object containing two fields: 'intent' (string: HIGH/MEDIUM/LOW) "
        "and 'confidence' (float between 0.0 and 1.0). No additional text."
    )

    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        if not self.api_key:
            raise ValueError("LLM_API_KEY environment variable is required")

        self.logger = AuditLogger.get_logger()
        self.client = httpx.AsyncClient(timeout=30.0)

        # API endpoint configurations
        if self.provider == "gemini":
            self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        elif self.provider == "deepseek":
            self.api_url = "https://api.deepseek.com/v1/chat/completions"
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    async def classify(self, post: Post) -> ClassifiedIntent:
        """
        Send a single post to the LLM and return a validated classification.
        """
        user_message = f"Post content: {post.content}\nSource: {post.source.value}\nTimestamp: {post.timestamp.isoformat()}"
        self.logger.debug(f"Classifying post {post.id} from {post.source.value}")

        try:
            if self.provider == "gemini":
                response = await self._call_gemini(user_message)
            else:  # deepseek (OpenAI-compatible)
                response = await self._call_deepseek(user_message)

            # Parse JSON response from LLM
            intent_data = self._parse_llm_response(response)
            intent_level = IntentLevel(intent_data["intent"].upper())
            confidence = float(intent_data["confidence"])

            classified = ClassifiedIntent(
                post=post,
                intent_level=intent_level,
                confidence_score=confidence,
                llm_response_raw=json.dumps(intent_data)
            )

            self.logger.info(
                f"Classification result | PostId={post.id} | Intent={intent_level.value} | "
                f"Confidence={confidence:.2f} | Provider={self.provider}"
            )
            return classified

        except Exception as e:
            self.logger.error(f"LLM classification failed for post {post.id}: {e}")
            # Fallback: assign LOW intent with zero confidence
            return ClassifiedIntent(
                post=post,
                intent_level=IntentLevel.LOW,
                confidence_score=0.0,
                llm_response_raw=f"ERROR: {str(e)}"
            )

    async def _call_gemini(self, user_message: str) -> Dict[str, Any]:
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": self.SYSTEM_PROMPT + "\n\n" + user_message}
                    ]
                }
            ]
        }
        resp = await self.client.post(self.api_url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        # Extract text from Gemini response
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except (KeyError, json.JSONDecodeError) as e:
            self.logger.error(f"Gemini response parsing error: {data}")
            raise ValueError(f"Invalid Gemini response format: {e}")

    async def _call_deepseek(self, user_message: str) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }
        resp = await self.client.post(self.api_url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            self.logger.error(f"DeepSeek response parsing error: {data}")
            raise ValueError(f"Invalid DeepSeek response format: {e}")

    @staticmethod
    def _parse_llm_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that LLM returned expected fields."""
        intent = response.get("intent", "LOW").upper()
        if intent not in ["HIGH", "MEDIUM", "LOW"]:
            intent = "LOW"
        confidence = response.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 0.0
        return {"intent": intent, "confidence": confidence}

    async def close(self):
        await self.client.aclose()


# -----------------------------------------------------------------------------
# Asynchronous Factory Pattern for Data Sources
# -----------------------------------------------------------------------------
class DataSourceScanner(ABC):
    """Abstract base class for a social media source scanner."""

    def __init__(self, source_type: SourceType, rate_limit_seconds: float):
        self.source_type = source_type
        self.rate_limit = rate_limit_seconds
        self.logger = AuditLogger.get_logger()

    @abstractmethod
    async def scan(self) -> AsyncGenerator[Post, None]:
        """
        Asynchronously yield posts from the source.
        Implementations simulate or connect to real APIs.
        """
        pass

class TwitterScanner(DataSourceScanner):
    """Simulated scanner for X/Twitter."""

    # Mock post templates reflecting common financial/security topics
    MOCK_POSTS = [
        "Bitcoin just crashed 20% in 10 minutes, what is happening?",
        "SEC announces new stablecoin regulations effective next week.",
        "Someone posted a phishing link pretending to be from our exchange.",
        "I think the market will go up tomorrow, nothing serious.",
        "URGENT: Large suspicious transfer detected on chain. Immediate freeze required.",
        "Just sharing my daily trading P&L: +2%.",
        "New exploit discovered in smart contract library XYZ, all devs must patch now.",
        "Routine maintenance announcement for the platform.",
    ]

    async def scan(self) -> AsyncGenerator[Post, None]:
        while True:
            content = random.choice(self.MOCK_POSTS)
            # Add some randomness to simulate different users
            post = Post(
                source=self.source_type,
                content=content,
                metadata={"simulated_user": f"user_{random.randint(1,1000)}"}
            )
            self.logger.debug(f"Twitter scanner generated post {post.id}")
            yield post
            await asyncio.sleep(self.rate_limit)

class RedditScanner(DataSourceScanner):
    """Simulated scanner for Reddit."""

    MOCK_POSTS = [
        "[Serious] Our subreddit is being brigaded with spam links.",
        "ELI5: Why does the price move after FOMC meetings?",
        "MOD NOTE: New rule against posting referral codes - effective immediately.",
        "I lost access to my 2FA, recovery process not working, urgent help needed.",
        "Has anyone used the new API for automated trading?",
        "Critical: Exchange wallet drained. Check your balances now.",
        "Just a meme about crypto volatility.",
        "Security audit report released for project X. No major issues.",
    ]

    async def scan(self) -> AsyncGenerator[Post, None]:
        while True:
            content = random.choice(self.MOCK_POSTS)
            post = Post(
                source=self.source_type,
                content=content,
                metadata={"subreddit": "cryptocurrency", "upvotes": random.randint(0, 500)}
            )
            self.logger.debug(f"Reddit scanner generated post {post.id}")
            yield post
            await asyncio.sleep(self.rate_limit)

class ScannerFactory:
    """Factory that creates appropriate scanner instances for each source."""

    @staticmethod
    def create_scanner(source_name: str, rate_limit: float = 5.0) -> DataSourceScanner:
        source_name = source_name.lower()
        if source_name == "twitter" or source_name == "x":
            return TwitterScanner(SourceType.TWITTER, rate_limit)
        elif source_name == "reddit":
            return RedditScanner(SourceType.REDDIT, rate_limit)
        else:
            raise ValueError(f"Unsupported source: {source_name}")


# -----------------------------------------------------------------------------
# Polyglot Bridge: Handoff to Rust/Node.js Microservice
# -----------------------------------------------------------------------------
class PolyglotBridge:
    """
    Demonstrates asynchronous handoff of classified intents to an external
    high-performance microservice (e.g., Rust with Actix, Node.js with Fastify).
    Uses HTTP POST with retry logic and structured logging.
    """

    def __init__(self, microservice_url: Optional[str] = None):
        self.microservice_url = microservice_url or os.getenv("MICROSERVICE_URL", "http://localhost:8080/intents")
        self.logger = AuditLogger.get_logger()
        self.client = httpx.AsyncClient(timeout=5.0)
        self.max_retries = 3
        self.retry_delay = 1.0

    async def dispatch(self, classified_intents: List[ClassifiedIntent]) -> bool:
        """
        Send a batch of classified intents to the microservice.
        Returns True if at least one attempt succeeded, False otherwise.
        """
        if not classified_intents:
            self.logger.debug("No intents to dispatch")
            return True

        payload = BridgePayload(classified_intents=classified_intents)
        request_id = payload.request_id
        self.logger.info(f"Dispatching {len(classified_intents)} intents to microservice | RequestId={request_id}")

        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(
                    self.microservice_url,
                    json=payload.dict(exclude_none=True, by_alias=False),
                    headers={"Content-Type": "application/json", "X-Request-Id": request_id}
                )
                response.raise_for_status()
                self.logger.info(f"Microservice acknowledged | RequestId={request_id} | Status={response.status_code}")
                return True

            except httpx.HTTPStatusError as e:
                self.logger.warning(
                    f"Microservice HTTP error | Attempt={attempt+1} | RequestId={request_id} | "
                    f"Status={e.response.status_code} | Detail={e.response.text[:200]}"
                )
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                self.logger.warning(f"Microservice connection error | Attempt={attempt+1} | RequestId={request_id} | Error={type(e).__name__}")
            except Exception as e:
                self.logger.error(f"Unexpected dispatch error | RequestId={request_id} | Error={e}")

            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (2 ** attempt))  # exponential backoff

        self.logger.error(f"Failed to dispatch after {self.max_retries} attempts | RequestId={request_id}")
        return False

    async def close(self):
        await self.client.aclose()


# -----------------------------------------------------------------------------
# Main Orchestrator
# -----------------------------------------------------------------------------
class AutonomousIntentMonitor:
    """
    Orchestrates multiple source scanners, LLM classification, polyglot bridge,
    and graceful shutdown.
    """

    def __init__(self):
        self.logger = AuditLogger.get_logger()
        self.rate_limit = float(os.getenv("SIMULATION_RATE_LIMIT", "5.0"))
        self.scanner_factory = ScannerFactory()
        self.classifier: Optional[IntentClassifier] = None
        self.bridge: Optional[PolyglotBridge] = None
        self._shutdown_event = asyncio.Event()
        self._active_tasks: List[asyncio.Task] = []
        self._batch_queue: asyncio.Queue = None
        self._batch_size = 5
        self._batch_interval = 10.0  # seconds

    async def _scan_and_enqueue(self, source_name: str):
        """Task: scan a single source and push posts into classification queue."""
        scanner = self.scanner_factory.create_scanner(source_name, self.rate_limit)
        self.logger.info(f"Started scanner for {source_name}")
        async for post in scanner.scan():
            if self._shutdown_event.is_set():
                break
            await self._batch_queue.put(post)

    async def _classify_worker(self):
        """Worker that pulls posts from queue, classifies them, and accumulates batches."""
        batch = []
        last_flush = asyncio.get_event_loop().time()

        while not self._shutdown_event.is_set() or not self._batch_queue.empty():
            try:
                # Wait for a post, but also check flush interval
                post = await asyncio.wait_for(self._batch_queue.get(), timeout=1.0)
                classified = await self.classifier.classify(post)
                batch.append(classified)

                if len(batch) >= self._batch_size:
                    await self._flush_batch(batch)
                    batch = []
                    last_flush = asyncio.get_event_loop().time()
                else:
                    # Check if it's time to flush based on interval
                    now = asyncio.get_event_loop().time()
                    if now - last_flush >= self._batch_interval and batch:
                        await self._flush_batch(batch)
                        batch = []
                        last_flush = now

            except asyncio.TimeoutError:
                # No post available, flush any pending batch if interval passed
                now = asyncio.get_event_loop().time()
                if batch and (now - last_flush >= self._batch_interval):
                    await self._flush_batch(batch)
                    batch = []
                    last_flush = now
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception(f"Classification worker error: {e}")

        # Final flush on shutdown
        if batch:
            await self._flush_batch(batch)

    async def _flush_batch(self, batch: List[ClassifiedIntent]):
        """Send batch to polyglot bridge and log audit trail."""
        if not batch:
            return

        # Audit log: each classified intent is logged individually for traceability
        for ci in batch:
            self.logger.info(
                f"AUDIT | PostId={ci.post.id} | Source={ci.post.source.value} | "
                f"Intent={ci.intent_level.value} | Confidence={ci.confidence_score:.2f}"
            )

        # Dispatch via polyglot bridge
        success = await self.bridge.dispatch(batch)
        if not success:
            self.logger.warning(f"Batch of {len(batch)} intents could not be delivered to microservice")
        else:
            self.logger.debug(f"Batch of {len(batch)} intents successfully dispatched")

    async def shutdown(self, sig: signal.Signals):
        """Graceful shutdown handler."""
        self.logger.info(f"Received shutdown signal {sig.name}. Initiating graceful shutdown...")
        self._shutdown_event.set()

        # Cancel all active tasks
        for task in self._active_tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(*self._active_tasks, return_exceptions=True)

        if self.classifier:
            await self.classifier.close()
        if self.bridge:
            await self.bridge.close()

        self.logger.info("Autonomous Intent Monitor terminated cleanly")
        sys.exit(0)

    async def run(self):
        """Start the monitor."""
        self.logger.info("Starting Autonomous Intent Monitor")

        # Initialize components
        self.classifier = IntentClassifier()
        self.bridge = PolyglotBridge()
        self._batch_queue = asyncio.Queue(maxsize=100)

        # Define sources to monitor
        sources = ["twitter", "reddit"]

        # Create and track tasks
        for src in sources:
            task = asyncio.create_task(self._scan_and_enqueue(src))
            self._active_tasks.append(task)

        # Classification workers (one worker is sufficient for demo; can scale)
        worker = asyncio.create_task(self._classify_worker())
        self._active_tasks.append(worker)

        # Register signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.shutdown(s)))

        # Wait for shutdown event
        await self._shutdown_event.wait()


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
async def main():
    monitor = AutonomousIntentMonitor()
    await monitor.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Fallback in case signal handler didn't catch
        print("\nInterrupt received. Exiting.")
        sys.exit(0)
