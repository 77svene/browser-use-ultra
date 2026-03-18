"""
Resilient Network Layer for Browser Use
Production-grade network layer with circuit breakers, exponential backoff,
request coalescing, and automatic CDP session recovery.
"""

import asyncio
import logging
import time
import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import inspect

from ..exceptions import BrowserError, NetworkError, CircuitBreakerError

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_requests: int = 3
    success_threshold: int = 2


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    base_delay: float = 0.1
    max_delay: float = 10.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class RequestRecord:
    """Record of an in-flight or completed request."""
    request_id: str
    domain: str
    method: str
    params: Dict[str, Any]
    future: asyncio.Future
    created_at: float = field(default_factory=time.monotonic)
    attempts: int = 0
    last_attempt: Optional[float] = None


class CircuitBreaker:
    """Circuit breaker for a specific domain."""
    
    def __init__(self, domain: str, config: CircuitBreakerConfig):
        self.domain = domain
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_requests = 0
        self._lock = asyncio.Lock()
    
    async def before_request(self) -> bool:
        """Check if request can proceed. Returns True if allowed."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self.last_failure_time and \
                   time.monotonic() - self.last_failure_time > self.config.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_requests = 0
                    logger.info(f"Circuit {self.domain} transitioning to HALF_OPEN")
                else:
                    return False
            
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_requests >= self.config.half_open_max_requests:
                    return False
                self.half_open_requests += 1
            
            return True
    
    async def record_success(self):
        """Record a successful request."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._reset()
                    logger.info(f"Circuit {self.domain} recovered to CLOSED")
            else:
                self.failure_count = max(0, self.failure_count - 1)
    
    async def record_failure(self):
        """Record a failed request."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()
            
            if self.state == CircuitState.HALF_OPEN:
                self._trip()
                logger.warning(f"Circuit {self.domain} failed during HALF_OPEN, tripping to OPEN")
            elif self.failure_count >= self.config.failure_threshold:
                self._trip()
                logger.warning(f"Circuit {self.domain} tripped to OPEN after {self.failure_count} failures")
    
    def _trip(self):
        """Trip the circuit breaker to OPEN state."""
        self.state = CircuitState.OPEN
        self.success_count = 0
        self.half_open_requests = 0
    
    def _reset(self):
        """Reset the circuit breaker to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_requests = 0
        self.last_failure_time = None
    
    @property
    def is_available(self) -> bool:
        """Check if circuit allows requests."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if self.last_failure_time and \
               time.monotonic() - self.last_failure_time > self.config.recovery_timeout:
                return True
            return False
        else:  # HALF_OPEN
            return self.half_open_requests < self.config.half_open_max_requests


class RequestCoalescer:
    """Coalesces identical CDP requests to prevent duplicates."""
    
    def __init__(self):
        self._pending_requests: Dict[str, List[RequestRecord]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    def _make_request_key(self, domain: str, method: str, params: Dict[str, Any]) -> str:
        """Create a unique key for request deduplication."""
        # Sort params for consistent hashing
        param_str = str(sorted(params.items())) if params else ""
        return f"{domain}:{method}:{param_str}"
    
    async def add_request(self, record: RequestRecord) -> Tuple[bool, Optional[asyncio.Future]]:
        """
        Add a request to the coalescer.
        Returns (is_new, future_to_await).
        If is_new is False, future_to_await is the existing future to await.
        """
        key = self._make_request_key(record.domain, record.method, record.params)
        
        async with self._lock:
            if key in self._pending_requests:
                # Coalesce with existing request
                self._pending_requests[key].append(record)
                return False, self._pending_requests[key][0].future
            else:
                # New request
                self._pending_requests[key].append(record)
                return True, None
    
    async def complete_request(self, record: RequestRecord, result: Any = None, 
                               error: Optional[Exception] = None):
        """Complete a request and notify all waiters."""
        key = self._make_request_key(record.domain, record.method, record.params)
        
        async with self._lock:
            if key not in self._pending_requests:
                return
            
            records = self._pending_requests.pop(key)
            for rec in records:
                if not rec.future.done():
                    if error:
                        rec.future.set_exception(error)
                    else:
                        rec.future.set_result(result)


class ResilientCDPClient:
    """
    Resilient wrapper for CDP client with circuit breakers, retries, and coalescing.
    """
    
    def __init__(self, 
                 cdp_client_factory: Callable,
                 circuit_config: Optional[CircuitBreakerConfig] = None,
                 retry_config: Optional[RetryConfig] = None,
                 health_check_interval: float = 30.0,
                 session_recovery_timeout: float = 60.0):
        """
        Args:
            cdp_client_factory: Async callable that creates a new CDP client instance
            circuit_config: Configuration for circuit breakers
            retry_config: Configuration for retry logic
            health_check_interval: Seconds between health checks
            session_recovery_timeout: Max seconds to wait for session recovery
        """
        self._cdp_client_factory = cdp_client_factory
        self._circuit_config = circuit_config or CircuitBreakerConfig()
        self._retry_config = retry_config or RetryConfig()
        self._health_check_interval = health_check_interval
        self._session_recovery_timeout = session_recovery_timeout
        
        self._cdp_client = None
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._coalescer = RequestCoalescer()
        self._request_counter = 0
        self._lock = asyncio.Lock()
        self._session_lock = asyncio.Lock()
        self._is_connected = False
        self._last_health_check = 0
        self._health_check_task: Optional[asyncio.Task] = None
        self._domains_in_use: Set[str] = set()
        
        # Event callbacks
        self._on_disconnect_callbacks: List[Callable] = []
        self._on_reconnect_callbacks: List[Callable] = []
        
        logger.info("ResilientCDPClient initialized")
    
    async def connect(self):
        """Establish connection to CDP."""
        async with self._session_lock:
            if self._is_connected and self._cdp_client:
                return
            
            try:
                self._cdp_client = await self._cdp_client_factory()
                self._is_connected = True
                self._last_health_check = time.monotonic()
                
                # Start health check task
                if self._health_check_task is None or self._health_check_task.done():
                    self._health_check_task = asyncio.create_task(self._health_check_loop())
                
                logger.info("CDP connection established")
                
                # Notify reconnection
                for callback in self._on_reconnect_callbacks:
                    try:
                        if inspect.iscoroutinefunction(callback):
                            await callback()
                        else:
                            callback()
                    except Exception as e:
                        logger.error(f"Error in reconnect callback: {e}")
                        
            except Exception as e:
                self._is_connected = False
                logger.error(f"Failed to connect to CDP: {e}")
                raise NetworkError(f"CDP connection failed: {e}")
    
    async def disconnect(self):
        """Disconnect from CDP."""
        async with self._session_lock:
            if self._cdp_client:
                try:
                    # Try graceful disconnect
                    if hasattr(self._cdp_client, 'close'):
                        await self._cdp_client.close()
                    elif hasattr(self._cdp_client, 'disconnect'):
                        await self._cdp_client.disconnect()
                except Exception as e:
                    logger.warning(f"Error during CDP disconnect: {e}")
                finally:
                    self._cdp_client = None
                    self._is_connected = False
            
            # Cancel health check
            if self._health_check_task and not self._health_check_task.done():
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
    
    def _get_circuit_breaker(self, domain: str) -> CircuitBreaker:
        """Get or create circuit breaker for domain."""
        if domain not in self._circuit_breakers:
            self._circuit_breakers[domain] = CircuitBreaker(domain, self._circuit_config)
        return self._circuit_breakers[domain]
    
    async def send(self, 
                   domain: str, 
                   method: str, 
                   params: Optional[Dict[str, Any]] = None,
                   timeout: Optional[float] = None) -> Any:
        """
        Send a CDP command with resilience patterns.
        
        Args:
            domain: CDP domain (e.g., 'Page', 'Network')
            method: Method name
            params: Method parameters
            timeout: Request timeout
            
        Returns:
            CDP response
            
        Raises:
            CircuitBreakerError: If circuit is open
            NetworkError: If all retries fail
        """
        params = params or {}
        request_id = f"req_{self._request_counter}"
        self._request_counter += 1
        
        # Track domain usage
        self._domains_in_use.add(domain)
        
        # Create request record
        future = asyncio.get_event_loop().create_future()
        record = RequestRecord(
            request_id=request_id,
            domain=domain,
            method=method,
            params=params,
            future=future
        )
        
        # Try to coalesce request
        is_new, existing_future = await self._coalescer.add_request(record)
        if not is_new:
            # Wait for existing request
            return await existing_future
        
        # Execute with resilience
        try:
            result = await self._execute_with_resilience(record, timeout)
            await self._coalescer.complete_request(record, result=result)
            return result
        except Exception as e:
            await self._coalescer.complete_request(record, error=e)
            raise
    
    async def _execute_with_resilience(self, record: RequestRecord, timeout: Optional[float]) -> Any:
        """Execute request with retry and circuit breaker logic."""
        circuit = self._get_circuit_breaker(record.domain)
        last_exception = None
        
        for attempt in range(self._retry_config.max_retries + 1):
            record.attempts = attempt + 1
            record.last_attempt = time.monotonic()
            
            # Check circuit breaker
            if not await circuit.before_request():
                raise CircuitBreakerError(
                    f"Circuit breaker open for domain {record.domain}"
                )
            
            try:
                # Ensure connection
                await self._ensure_connected()
                
                # Execute request
                result = await self._send_cdp_request(
                    record.domain, record.method, record.params, timeout
                )
                
                # Record success
                await circuit.record_success()
                return result
                
            except (ConnectionError, TimeoutError, asyncio.TimeoutError) as e:
                last_exception = e
                await circuit.record_failure()
                
                # Check if we should retry
                if attempt < self._retry_config.max_retries:
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(
                        f"Request {record.request_id} failed (attempt {attempt + 1}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Request {record.request_id} failed after {attempt + 1} attempts: {e}"
                    )
                    
            except Exception as e:
                # Non-retryable error
                await circuit.record_failure()
                logger.error(f"Non-retryable error for {record.request_id}: {e}")
                raise
        
        # All retries exhausted
        raise NetworkError(
            f"Request {record.request_id} failed after {self._retry_config.max_retries + 1} attempts: {last_exception}"
        )
    
    async def _send_cdp_request(self, domain: str, method: str, 
                                params: Dict[str, Any], timeout: Optional[float]) -> Any:
        """Send request to CDP client."""
        if not self._cdp_client:
            raise NetworkError("CDP client not connected")
        
        # Build CDP command
        command = f"{domain}.{method}"
        
        try:
            # Different CDP clients have different interfaces
            if hasattr(self._cdp_client, 'send'):
                # Standard send method
                if timeout:
                    return await asyncio.wait_for(
                        self._cdp_client.send(command, params),
                        timeout=timeout
                    )
                else:
                    return await self._cdp_client.send(command, params)
                    
            elif hasattr(self._cdp_client, 'execute'):
                # Some clients use execute
                if timeout:
                    return await asyncio.wait_for(
                        self._cdp_client.execute(command, params),
                        timeout=timeout
                    )
                else:
                    return await self._cdp_client.execute(command, params)
                    
            elif hasattr(self._cdp_client, 'call'):
                # Call-based interface
                if timeout:
                    return await asyncio.wait_for(
                        self._cdp_client.call(command, params),
                        timeout=timeout
                    )
                else:
                    return await self._cdp_client.call(command, params)
                    
            else:
                # Try dynamic attribute access
                domain_obj = getattr(self._cdp_client, domain, None)
                if domain_obj and hasattr(domain_obj, method):
                    method_func = getattr(domain_obj, method)
                    if timeout:
                        return await asyncio.wait_for(
                            method_func(**params),
                            timeout=timeout
                        )
                    else:
                        return await method_func(**params)
                else:
                    raise NetworkError(f"Unsupported CDP client interface")
                    
        except asyncio.TimeoutError:
            raise TimeoutError(f"CDP request timed out after {timeout}s")
        except Exception as e:
            # Wrap connection errors
            if "connection" in str(e).lower() or "closed" in str(e).lower():
                raise ConnectionError(f"CDP connection error: {e}")
            raise
    
    async def _ensure_connected(self):
        """Ensure CDP connection is active, reconnect if needed."""
        if not self._is_connected or not self._cdp_client:
            logger.info("CDP not connected, attempting to reconnect")
            await self.connect()
            return
        
        # Quick health check
        try:
            # Try a simple CDP command
            if hasattr(self._cdp_client, 'send'):
                await asyncio.wait_for(
                    self._cdp_client.send("Browser.getVersion", {}),
                    timeout=2.0
                )
            else:
                # Skip health check if interface doesn't support it
                pass
        except Exception as e:
            logger.warning(f"CDP health check failed: {e}")
            await self.disconnect()
            await self.connect()
    
    async def _health_check_loop(self):
        """Background task for health monitoring."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                
                if not self._is_connected:
                    continue
                
                # Perform health check
                try:
                    await self._ensure_connected()
                    self._last_health_check = time.monotonic()
                except Exception as e:
                    logger.error(f"Health check failed: {e}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(5)  # Prevent tight loop on errors
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        delay = min(
            self._retry_config.base_delay * (self._retry_config.exponential_base ** attempt),
            self._retry_config.max_delay
        )
        
        if self._retry_config.jitter:
            # Add jitter (±25%)
            jitter = delay * 0.25 * (2 * random.random() - 1)
            delay += jitter
        
        return max(0, delay)
    
    async def batch_send(self, commands: List[Tuple[str, str, Dict[str, Any]]], 
                         timeout: Optional[float] = None) -> List[Any]:
        """
        Send multiple CDP commands with resilience.
        
        Args:
            commands: List of (domain, method, params) tuples
            timeout: Timeout for entire batch
            
        Returns:
            List of results in same order as commands
        """
        tasks = []
        for domain, method, params in commands:
            task = asyncio.create_task(
                self.send(domain, method, params, timeout)
            )
            tasks.append(task)
        
        # Wait for all with timeout
        if timeout:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to proper errors
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                domain, method, _ = commands[i]
                logger.error(f"Batch command {domain}.{method} failed: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def on_disconnect(self, callback: Callable):
        """Register callback for disconnect events."""
        self._on_disconnect_callbacks.append(callback)
    
    def on_reconnect(self, callback: Callable):
        """Register callback for reconnect events."""
        self._on_reconnect_callbacks.append(callback)
    
    async def get_circuit_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        status = {}
        for domain, circuit in self._circuit_breakers.items():
            status[domain] = {
                "state": circuit.state.value,
                "failure_count": circuit.failure_count,
                "success_count": circuit.success_count,
                "last_failure": circuit.last_failure_time,
                "is_available": circuit.is_available
            }
        return status
    
    async def reset_circuit(self, domain: str):
        """Manually reset a circuit breaker."""
        if domain in self._circuit_breakers:
            self._circuit_breakers[domain]._reset()
            logger.info(f"Circuit {domain} manually reset")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


class BrowserProcessManager:
    """Manages browser process health and recovery."""
    
    def __init__(self, 
                 browser_factory: Callable,
                 health_check_interval: float = 10.0,
                 max_restart_attempts: int = 3):
        self._browser_factory = browser_factory
        self._health_check_interval = health_check_interval
        self._max_restart_attempts = max_restart_attempts
        
        self._browser_process = None
        self._is_running = False
        self._restart_attempts = 0
        self._health_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # Callbacks
        self._on_process_died_callbacks: List[Callable] = []
        self._on_process_restarted_callbacks: List[Callable] = []
    
    async def start(self):
        """Start browser process and health monitoring."""
        async with self._lock:
            if self._is_running:
                return
            
            await self._start_browser()
            self._is_running = True
            
            # Start health monitoring
            self._health_task = asyncio.create_task(self._health_monitor())
    
    async def stop(self):
        """Stop browser process and monitoring."""
        async with self._lock:
            self._is_running = False
            
            if self._health_task:
                self._health_task.cancel()
                try:
                    await self._health_task
                except asyncio.CancelledError:
                    pass
            
            await self._stop_browser()
    
    async def _start_browser(self):
        """Start the browser process."""
        try:
            self._browser_process = await self._browser_factory()
            self._restart_attempts = 0
            logger.info("Browser process started")
        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            raise
    
    async def _stop_browser(self):
        """Stop the browser process."""
        if self._browser_process:
            try:
                if hasattr(self._browser_process, 'close'):
                    await self._browser_process.close()
                elif hasattr(self._browser_process, 'kill'):
                    await self._browser_process.kill()
                elif hasattr(self._browser_process, 'terminate'):
                    self._browser_process.terminate()
            except Exception as e:
                logger.warning(f"Error stopping browser: {e}")
            finally:
                self._browser_process = None
    
    async def _health_monitor(self):
        """Monitor browser process health."""
        while self._is_running:
            try:
                await asyncio.sleep(self._health_check_interval)
                
                if not await self._check_browser_health():
                    logger.warning("Browser process unhealthy, attempting restart")
                    await self._restart_browser()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _check_browser_health(self) -> bool:
        """Check if browser process is healthy."""
        if not self._browser_process:
            return False
        
        try:
            # Different checks depending on browser type
            if hasattr(self._browser_process, 'is_connected'):
                return await self._browser_process.is_connected()
            elif hasattr(self._browser_process, 'poll'):
                # Process-based browser
                return self._browser_process.poll() is None
            else:
                # Assume healthy if process exists
                return True
        except Exception:
            return False
    
    async def _restart_browser(self):
        """Restart the browser process."""
        if self._restart_attempts >= self._max_restart_attempts:
            logger.error(f"Max restart attempts ({self._max_restart_attempts}) reached")
            raise RuntimeError("Browser process restart failed")
        
        self._restart_attempts += 1
        
        # Notify process died
        for callback in self._on_process_died_callbacks:
            try:
                if inspect.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Error in process died callback: {e}")
        
        # Stop old process
        await self._stop_browser()
        
        # Start new process
        await self._start_browser()
        
        # Notify process restarted
        for callback in self._on_process_restarted_callbacks:
            try:
                if inspect.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Error in process restarted callback: {e}")
    
    def on_process_died(self, callback: Callable):
        """Register callback for when browser process dies."""
        self._on_process_died_callbacks.append(callback)
    
    def on_process_restarted(self, callback: Callable):
        """Register callback for when browser process is restarted."""
        self._on_process_restarted_callbacks.append(callback)
    
    @property
    def is_running(self) -> bool:
        """Check if browser process is running."""
        return self._is_running and self._browser_process is not None


class ResilientNetworkLayer:
    """
    Complete resilient network layer combining CDP client resilience
    with browser process management.
    """
    
    def __init__(self,
                 cdp_client_factory: Callable,
                 browser_factory: Optional[Callable] = None,
                 circuit_config: Optional[CircuitBreakerConfig] = None,
                 retry_config: Optional[RetryConfig] = None,
                 health_check_interval: float = 30.0,
                 browser_health_interval: float = 10.0):
        
        self._cdp_client = ResilientCDPClient(
            cdp_client_factory=cdp_client_factory,
            circuit_config=circuit_config,
            retry_config=retry_config,
            health_check_interval=health_check_interval
        )
        
        self._browser_manager = None
        if browser_factory:
            self._browser_manager = BrowserProcessManager(
                browser_factory=browser_factory,
                health_check_interval=browser_health_interval
            )
        
        # Wire up callbacks
        if self._browser_manager:
            self._browser_manager.on_process_died(self._on_browser_died)
            self._browser_manager.on_process_restarted(self._on_browser_restarted)
    
    async def start(self):
        """Start the network layer."""
        if self._browser_manager:
            await self._browser_manager.start()
        
        await self._cdp_client.connect()
    
    async def stop(self):
        """Stop the network layer."""
        await self._cdp_client.disconnect()
        
        if self._browser_manager:
            await self._browser_manager.stop()
    
    async def _on_browser_died(self):
        """Handle browser process death."""
        logger.warning("Browser process died, disconnecting CDP")
        await self._cdp_client.disconnect()
    
    async def _on_browser_restarted(self):
        """Handle browser process restart."""
        logger.info("Browser process restarted, reconnecting CDP")
        await self._cdp_client.connect()
    
    async def send(self, domain: str, method: str, 
                   params: Optional[Dict[str, Any]] = None,
                   timeout: Optional[float] = None) -> Any:
        """Send CDP command through resilient layer."""
        return await self._cdp_client.send(domain, method, params, timeout)
    
    async def batch_send(self, commands: List[Tuple[str, str, Dict[str, Any]]],
                         timeout: Optional[float] = None) -> List[Any]:
        """Send batch of CDP commands."""
        return await self._cdp_client.batch_send(commands, timeout)
    
    def on_disconnect(self, callback: Callable):
        """Register disconnect callback."""
        self._cdp_client.on_disconnect(callback)
    
    def on_reconnect(self, callback: Callable):
        """Register reconnect callback."""
        self._cdp_client.on_reconnect(callback)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the network layer."""
        status = {
            "cdp_connected": self._cdp_client._is_connected,
            "circuit_status": await self._cdp_client.get_circuit_status(),
            "domains_in_use": list(self._cdp_client._domains_in_use),
        }
        
        if self._browser_manager:
            status["browser_running"] = self._browser_manager.is_running
            status["browser_restart_attempts"] = self._browser_manager._restart_attempts
        
        return status
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Factory function for easy integration
def create_resilient_network_layer(
    cdp_client_factory: Callable,
    browser_factory: Optional[Callable] = None,
    **kwargs
) -> ResilientNetworkLayer:
    """
    Factory function to create a resilient network layer.
    
    Args:
        cdp_client_factory: Async callable that creates CDP client
        browser_factory: Optional async callable that creates browser process
        **kwargs: Additional configuration options
    
    Returns:
        Configured ResilientNetworkLayer instance
    """
    return ResilientNetworkLayer(
        cdp_client_factory=cdp_client_factory,
        browser_factory=browser_factory,
        **kwargs
    )