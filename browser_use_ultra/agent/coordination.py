"""
Parallel Agent Orchestration Engine for browser-use-ultra.
Enables multiple AI agents to operate concurrently with intelligent resource sharing,
conflict resolution, and task decomposition.
"""

import asyncio
import uuid
import time
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Callable, Awaitable
from collections import defaultdict
import json

from browser_use_ultra.agent.service import Agent
from browser_use_ultra.agent.views import AgentState, AgentStatus
from browser_use_ultra.actor.page import Page
from browser_use_ultra.agent.message_manager.service import MessageManager
from browser_use_ultra.agent.message_manager.views import Message

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Specialized agent types for different browser tasks."""
    NAVIGATION = auto()
    FORM_FILLING = auto()
    DATA_EXTRACTION = auto()
    INTERACTION = auto()
    MONITORING = auto()


class TaskPriority(Enum):
    """Task priority levels for workload balancing."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class ResourceType(Enum):
    """Shared resource types that require coordination."""
    BROWSER_CONTEXT = auto()
    LOGIN_STATE = auto()
    DOM_ELEMENT = auto()
    NETWORK = auto()
    LOCAL_STORAGE = auto()


@dataclass
class AgentCapability:
    """Defines what an agent can do."""
    agent_type: AgentType
    skills: Set[str] = field(default_factory=set)
    max_concurrent_tasks: int = 3
    performance_score: float = 1.0
    supported_pages: Set[str] = field(default_factory=set)


@dataclass
class TaskDefinition:
    """Defines a task to be executed by agents."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""
    description: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    required_capabilities: Set[str] = field(default_factory=set)
    target_url: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: float = 300.0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    agent_id: str
    status: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0
    completed_at: float = field(default_factory=time.time)


@dataclass
class ResourceLock:
    """Represents a lock on a shared resource."""
    resource_type: ResourceType
    resource_id: str
    owner_agent_id: str
    acquired_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    is_shared: bool = False


@dataclass
class DOMVersion:
    """Tracks DOM modifications for optimistic concurrency control."""
    page_url: str
    element_selector: str
    version: int = 0
    last_modified_by: Optional[str] = None
    last_modified_at: float = field(default_factory=time.time)


class AgentCommunicationBus:
    """Message-passing system for agent communication."""
    
    def __init__(self):
        self.message_queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self.subscribers: Dict[str, Set[str]] = defaultdict(set)
        self.message_history: Dict[str, List[Message]] = defaultdict(list)
    
    async def send_message(self, sender_id: str, receiver_id: str, message_type: str, 
                          content: Dict[str, Any], priority: int = 0):
        """Send a message from one agent to another."""
        message = Message(
            id=str(uuid.uuid4()),
            sender=sender_id,
            receiver=receiver_id,
            type=message_type,
            content=content,
            priority=priority,
            timestamp=time.time()
        )
        
        await self.message_queues[receiver_id].put(message)
        self.message_history[receiver_id].append(message)
        
        # Notify subscribers
        for subscriber in self.subscribers.get(receiver_id, set()):
            if subscriber != sender_id:
                await self.message_queues[subscriber].put(message)
        
        return message.id
    
    async def broadcast(self, sender_id: str, message_type: str, content: Dict[str, Any]):
        """Broadcast a message to all agents."""
        for agent_id in list(self.message_queues.keys()):
            if agent_id != sender_id:
                await self.send_message(sender_id, agent_id, message_type, content)
    
    def subscribe(self, agent_id: str, target_agent_id: str):
        """Subscribe to messages from another agent."""
        self.subscribers[target_agent_id].add(agent_id)
    
    def unsubscribe(self, agent_id: str, target_agent_id: str):
        """Unsubscribe from messages."""
        self.subscribers[target_agent_id].discard(agent_id)


class ResourceManager:
    """Manages shared resources and handles conflict resolution."""
    
    def __init__(self):
        self.locks: Dict[str, ResourceLock] = {}
        self.dom_versions: Dict[str, DOMVersion] = {}
        self.login_states: Dict[str, Dict[str, Any]] = {}
        self.resource_waiters: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
    
    async def acquire_resource(self, agent_id: str, resource_type: ResourceType, 
                              resource_id: str, timeout: float = 30.0, 
                              shared: bool = False) -> bool:
        """Acquire a lock on a shared resource."""
        lock_key = f"{resource_type.value}:{resource_id}"
        
        # Check if resource is already locked
        if lock_key in self.locks:
            existing_lock = self.locks[lock_key]
            
            # If shared lock requested and existing is shared, allow
            if shared and existing_lock.is_shared:
                return True
            
            # Wait for resource to be released
            try:
                await asyncio.wait_for(
                    self.resource_waiters[lock_key].put(agent_id),
                    timeout=timeout
                )
                await asyncio.wait_for(
                    self.resource_waiters[lock_key].get(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                return False
        
        # Acquire the lock
        self.locks[lock_key] = ResourceLock(
            resource_type=resource_type,
            resource_id=resource_id,
            owner_agent_id=agent_id,
            is_shared=shared
        )
        
        return True
    
    def release_resource(self, agent_id: str, resource_type: ResourceType, 
                        resource_id: str):
        """Release a resource lock."""
        lock_key = f"{resource_type.value}:{resource_id}"
        
        if lock_key in self.locks and self.locks[lock_key].owner_agent_id == agent_id:
            del self.locks[lock_key]
            
            # Notify waiters
            if lock_key in self.resource_waiters and not self.resource_waiters[lock_key].empty():
                try:
                    self.resource_waiters[lock_key].get_nowait()
                except asyncio.QueueEmpty:
                    pass
    
    def check_dom_version(self, page_url: str, element_selector: str, 
                         expected_version: int) -> bool:
        """Check DOM version for optimistic concurrency control."""
        version_key = f"{page_url}:{element_selector}"
        
        if version_key not in self.dom_versions:
            self.dom_versions[version_key] = DOMVersion(
                page_url=page_url,
                element_selector=element_selector
            )
        
        return self.dom_versions[version_key].version == expected_version
    
    def update_dom_version(self, page_url: str, element_selector: str, 
                          agent_id: str) -> int:
        """Update DOM version after modification."""
        version_key = f"{page_url}:{element_selector}"
        
        if version_key not in self.dom_versions:
            self.dom_versions[version_key] = DOMVersion(
                page_url=page_url,
                element_selector=element_selector
            )
        
        version = self.dom_versions[version_key]
        version.version += 1
        version.last_modified_by = agent_id
        version.last_modified_at = time.time()
        
        return version.version
    
    def get_login_state(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get shared login state for a domain."""
        return self.login_states.get(domain)
    
    def update_login_state(self, domain: str, state: Dict[str, Any], 
                          agent_id: str):
        """Update shared login state."""
        self.login_states[domain] = {
            **state,
            "updated_by": agent_id,
            "updated_at": time.time()
        }


class WorkloadBalancer:
    """Dynamically balances workload across agents."""
    
    def __init__(self):
        self.agent_workloads: Dict[str, int] = defaultdict(int)
        self.agent_performance: Dict[str, float] = defaultdict(lambda: 1.0)
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
    
    def update_workload(self, agent_id: str, delta: int):
        """Update agent workload."""
        self.agent_workloads[agent_id] = max(0, self.agent_workloads[agent_id] + delta)
    
    def get_best_agent(self, task: TaskDefinition, 
                      available_agents: List[AgentCapability]) -> Optional[str]:
        """Select the best agent for a task based on capabilities and workload."""
        if not available_agents:
            return None
        
        # Filter agents by required capabilities
        capable_agents = []
        for agent_cap in available_agents:
            if task.required_capabilities.issubset(agent_cap.skills):
                capable_agents.append(agent_cap)
        
        if not capable_agents:
            return None
        
        # Score agents based on workload and performance
        scored_agents = []
        for agent_cap in capable_agents:
            workload = self.agent_workloads.get(agent_cap.agent_type.name, 0)
            performance = self.agent_performance.get(agent_cap.agent_type.name, 1.0)
            
            # Lower score is better
            score = (workload / agent_cap.max_concurrent_tasks) / performance
            scored_agents.append((agent_cap.agent_type.name, score))
        
        # Select agent with lowest score
        scored_agents.sort(key=lambda x: x[1])
        return scored_agents[0][0]
    
    def assign_task(self, task_id: str, agent_id: str):
        """Record task assignment."""
        self.task_assignments[task_id] = agent_id
        self.update_workload(agent_id, 1)
    
    def complete_task(self, task_id: str, agent_id: str, 
                     success: bool, execution_time: float):
        """Record task completion and update performance metrics."""
        if task_id in self.task_assignments:
            del self.task_assignments[task_id]
        
        self.update_workload(agent_id, -1)
        
        # Update performance score (exponential moving average)
        current_perf = self.agent_performance[agent_id]
        success_rate = 1.0 if success else 0.0
        self.agent_performance[agent_id] = (current_perf * 0.7) + (success_rate * 0.3)


class SpecializedAgent(Agent):
    """Extended Agent with specialization and coordination capabilities."""
    
    def __init__(self, agent_id: str, agent_type: AgentType, 
                 capabilities: List[str], orchestrator: 'Orchestrator',
                 **kwargs):
        super().__init__(**kwargs)
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = set(capabilities)
        self.orchestrator = orchestrator
        self.current_tasks: Dict[str, TaskDefinition] = {}
        self.message_queue = asyncio.Queue()
        self.running = False
    
    async def start(self):
        """Start the agent."""
        self.running = True
        await asyncio.gather(
            self._process_messages(),
            self._execute_tasks()
        )
    
    async def stop(self):
        """Stop the agent."""
        self.running = False
    
    async def _process_messages(self):
        """Process incoming messages."""
        while self.running:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                await self._handle_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Agent {self.agent_id} message processing error: {e}")
    
    async def _handle_message(self, message: Message):
        """Handle incoming message."""
        if message.type == "task_assignment":
            task = TaskDefinition(**message.content["task"])
            await self._accept_task(task)
        elif message.type == "resource_request":
            await self._handle_resource_request(message)
        elif message.type == "dom_conflict":
            await self._handle_dom_conflict(message)
        elif message.type == "coordination":
            await self._handle_coordination(message)
    
    async def _accept_task(self, task: TaskDefinition):
        """Accept a task assignment."""
        self.current_tasks[task.task_id] = task
        await self.orchestrator.resource_manager.acquire_resource(
            self.agent_id, ResourceType.BROWSER_CONTEXT, task.task_id
        )
        
        # Notify orchestrator
        await self.orchestrator.communication_bus.send_message(
            self.agent_id, "orchestrator", "task_accepted",
            {"task_id": task.task_id}
        )
    
    async def _execute_tasks(self):
        """Execute assigned tasks."""
        while self.running:
            for task_id, task in list(self.current_tasks.items()):
                try:
                    result = await self._execute_task(task)
                    await self._report_task_completion(task, result)
                    del self.current_tasks[task_id]
                except Exception as e:
                    await self._report_task_failure(task, str(e))
            
            await asyncio.sleep(0.1)
    
    async def _execute_task(self, task: TaskDefinition) -> TaskResult:
        """Execute a single task with conflict detection."""
        start_time = time.time()
        
        # Check for DOM conflicts before modification
        if "element_selector" in task.parameters:
            selector = task.parameters["element_selector"]
            page_url = task.target_url or "current"
            
            # Get current DOM version
            version_key = f"{page_url}:{selector}"
            current_version = self.orchestrator.resource_manager.dom_versions.get(
                version_key, DOMVersion(page_url, selector)
            ).version
            
            # Execute task
            try:
                result = await self._perform_task_action(task)
                
                # Update DOM version on success
                if result.status == "success":
                    self.orchestrator.resource_manager.update_dom_version(
                        page_url, selector, self.agent_id
                    )
                
                return TaskResult(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    status=result.status,
                    data=result.data,
                    execution_time=time.time() - start_time
                )
            except Exception as e:
                # Check if it's a DOM conflict
                if "stale element" in str(e).lower() or "version mismatch" in str(e).lower():
                    await self._request_dom_retry(task, current_version)
                raise
        else:
            # Execute non-DOM task
            result = await self._perform_task_action(task)
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=result.status,
                data=result.data,
                execution_time=time.time() - start_time
            )
    
    async def _perform_task_action(self, task: TaskDefinition) -> TaskResult:
        """Perform the actual task action based on agent type."""
        # This would be implemented by specialized agent subclasses
        # For now, return a placeholder
        return TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status="success",
            data={"message": "Task executed"}
        )
    
    async def _request_dom_retry(self, task: TaskDefinition, expected_version: int):
        """Request retry due to DOM conflict."""
        await self.orchestrator.communication_bus.send_message(
            self.agent_id, "orchestrator", "dom_conflict",
            {
                "task_id": task.task_id,
                "expected_version": expected_version,
                "selector": task.parameters.get("element_selector"),
                "page_url": task.target_url
            }
        )
    
    async def _report_task_completion(self, task: TaskDefinition, result: TaskResult):
        """Report task completion to orchestrator."""
        await self.orchestrator.communication_bus.send_message(
            self.agent_id, "orchestrator", "task_completed",
            {"task_id": task.task_id, "result": result.__dict__}
        )
    
    async def _report_task_failure(self, task: TaskDefinition, error: str):
        """Report task failure to orchestrator."""
        await self.orchestrator.communication_bus.send_message(
            self.agent_id, "orchestrator", "task_failed",
            {"task_id": task.task_id, "error": error}
        )
    
    async def _handle_resource_request(self, message: Message):
        """Handle resource request from another agent."""
        resource_type = ResourceType(message.content["resource_type"])
        resource_id = message.content["resource_id"]
        requester_id = message.sender
        
        # Check if we can share the resource
        can_share = resource_type in [ResourceType.LOGIN_STATE, ResourceType.NETWORK]
        
        if can_share:
            # Grant shared access
            await self.orchestrator.communication_bus.send_message(
                self.agent_id, requester_id, "resource_granted",
                {
                    "resource_type": resource_type.value,
                    "resource_id": resource_id,
                    "shared": True
                }
            )
    
    async def _handle_dom_conflict(self, message: Message):
        """Handle DOM conflict notification."""
        task_id = message.content["task_id"]
        if task_id in self.current_tasks:
            # Requeue the task for retry
            task = self.current_tasks[task_id]
            await self.orchestrator.requeue_task(task)
            del self.current_tasks[task_id]
    
    async def _handle_coordination(self, message: Message):
        """Handle coordination messages."""
        # Implement coordination logic based on message content
        pass


class NavigationAgent(SpecializedAgent):
    """Specialized agent for page navigation."""
    
    def __init__(self, **kwargs):
        capabilities = ["navigation", "url_handling", "page_loading", "history_management"]
        super().__init__(agent_type=AgentType.NAVIGATION, capabilities=capabilities, **kwargs)
    
    async def _perform_task_action(self, task: TaskDefinition) -> TaskResult:
        """Perform navigation task."""
        url = task.parameters.get("url")
        if not url:
            raise ValueError("URL required for navigation task")
        
        # Use existing page navigation
        page = await self._get_page()
        await page.goto(url)
        
        return TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status="success",
            data={"url": url, "title": await page.title()}
        )


class FormFillingAgent(SpecializedAgent):
    """Specialized agent for form filling."""
    
    def __init__(self, **kwargs):
        capabilities = ["form_filling", "input_handling", "validation", "submission"]
        super().__init__(agent_type=AgentType.FORM_FILLING, capabilities=capabilities, **kwargs)
    
    async def _perform_task_action(self, task: TaskDefinition) -> TaskResult:
        """Perform form filling task."""
        form_data = task.parameters.get("form_data", {})
        selector = task.parameters.get("form_selector", "form")
        
        page = await self._get_page()
        
        # Fill form fields
        for field_name, value in form_data.items():
            field_selector = f"{selector} [name='{field_name}']"
            await page.fill(field_selector, value)
        
        # Submit if requested
        if task.parameters.get("submit", False):
            submit_selector = f"{selector} [type='submit']"
            await page.click(submit_selector)
        
        return TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status="success",
            data={"filled_fields": list(form_data.keys())}
        )


class DataExtractionAgent(SpecializedAgent):
    """Specialized agent for data extraction."""
    
    def __init__(self, **kwargs):
        capabilities = ["data_extraction", "parsing", "scraping", "analysis"]
        super().__init__(agent_type=AgentType.DATA_EXTRACTION, capabilities=capabilities, **kwargs)
    
    async def _perform_task_action(self, task: TaskDefinition) -> TaskResult:
        """Perform data extraction task."""
        selectors = task.parameters.get("selectors", {})
        page = await self._get_page()
        
        extracted_data = {}
        for name, selector in selectors.items():
            try:
                elements = await page.query_selector_all(selector)
                extracted_data[name] = [
                    await element.text_content() for element in elements
                ]
            except Exception as e:
                extracted_data[name] = f"Error: {str(e)}"
        
        return TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status="success",
            data={"extracted_data": extracted_data}
        )


class Orchestrator:
    """Central orchestrator for parallel agent management."""
    
    def __init__(self, max_agents: int = 10):
        self.max_agents = max_agents
        self.agents: Dict[str, SpecializedAgent] = {}
        self.agent_capabilities: Dict[str, AgentCapability] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.failed_tasks: Dict[str, TaskDefinition] = {}
        self.retry_counts: Dict[str, int] = defaultdict(int)
        
        self.communication_bus = AgentCommunicationBus()
        self.resource_manager = ResourceManager()
        self.workload_balancer = WorkloadBalancer()
        
        self.running = False
        self._orchestrator_id = "orchestrator"
    
    async def start(self):
        """Start the orchestrator."""
        self.running = True
        logger.info("Starting Parallel Agent Orchestrator")
        
        # Start message processing
        asyncio.create_task(self._process_messages())
        
        # Start task assignment loop
        asyncio.create_task(self._assign_tasks())
    
    async def stop(self):
        """Stop the orchestrator and all agents."""
        self.running = False
        
        # Stop all agents
        for agent in self.agents.values():
            await agent.stop()
        
        logger.info("Orchestrator stopped")
    
    async def register_agent(self, agent: SpecializedAgent, 
                            capabilities: List[str]):
        """Register an agent with the orchestrator."""
        if len(self.agents) >= self.max_agents:
            raise ValueError(f"Maximum agent limit ({self.max_agents}) reached")
        
        agent_id = agent.agent_id
        self.agents[agent_id] = agent
        
        # Create capability profile
        self.agent_capabilities[agent_id] = AgentCapability(
            agent_type=agent.agent_type,
            skills=agent.capabilities.union(set(capabilities)),
            max_concurrent_tasks=3,
            performance_score=1.0
        )
        
        # Subscribe to orchestrator messages
        self.communication_bus.subscribe(agent_id, self._orchestrator_id)
        
        logger.info(f"Registered agent {agent_id} of type {agent.agent_type.name}")
        
        # Start agent
        asyncio.create_task(agent.start())
    
    async def unregister_agent(self, agent_id: str):
        """Unregister an agent."""
        if agent_id in self.agents:
            await self.agents[agent_id].stop()
            del self.agents[agent_id]
            del self.agent_capabilities[agent_id]
            self.communication_bus.unsubscribe(agent_id, self._orchestrator_id)
            logger.info(f"Unregistered agent {agent_id}")
    
    async def submit_task(self, task: TaskDefinition):
        """Submit a task for execution."""
        # Add to priority queue (lower priority number = higher priority)
        await self.task_queue.put((task.priority.value, task))
        logger.info(f"Submitted task {task.task_id}: {task.description}")
    
    async def submit_tasks(self, tasks: List[TaskDefinition]):
        """Submit multiple tasks."""
        for task in tasks:
            await self.submit_task(task)
    
    async def decompose_task(self, high_level_task: Dict[str, Any]) -> List[TaskDefinition]:
        """Decompose a high-level task into subtasks."""
        # This is a simplified decomposition - in practice, this would use
        # more sophisticated planning algorithms
        subtasks = []
        
        task_type = high_level_task.get("type", "")
        
        if task_type == "web_scraping":
            # Decompose into navigation, extraction, and processing
            subtasks.extend([
                TaskDefinition(
                    task_type="navigation",
                    description=f"Navigate to {high_level_task['url']}",
                    priority=TaskPriority.HIGH,
                    required_capabilities={"navigation"},
                    target_url=high_level_task["url"],
                    parameters={"url": high_level_task["url"]}
                ),
                TaskDefinition(
                    task_type="data_extraction",
                    description="Extract data from page",
                    priority=TaskPriority.MEDIUM,
                    required_capabilities={"data_extraction"},
                    target_url=high_level_task["url"],
                    parameters={"selectors": high_level_task.get("selectors", {})},
                    dependencies=[subtasks[0].task_id] if subtasks else []
                )
            ])
        
        elif task_type == "form_submission":
            # Decompose into navigation and form filling
            subtasks.extend([
                TaskDefinition(
                    task_type="navigation",
                    description=f"Navigate to form at {high_level_task['url']}",
                    priority=TaskPriority.HIGH,
                    required_capabilities={"navigation"},
                    target_url=high_level_task["url"],
                    parameters={"url": high_level_task["url"]}
                ),
                TaskDefinition(
                    task_type="form_filling",
                    description="Fill and submit form",
                    priority=TaskPriority.MEDIUM,
                    required_capabilities={"form_filling"},
                    target_url=high_level_task["url"],
                    parameters={
                        "form_data": high_level_task.get("form_data", {}),
                        "submit": high_level_task.get("submit", True)
                    },
                    dependencies=[subtasks[0].task_id] if subtasks else []
                )
            ])
        
        return subtasks
    
    async def _process_messages(self):
        """Process incoming messages from agents."""
        orchestrator_queue = self.communication_bus.message_queues[self._orchestrator_id]
        
        while self.running:
            try:
                message = await asyncio.wait_for(
                    orchestrator_queue.get(),
                    timeout=1.0
                )
                await self._handle_agent_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Orchestrator message processing error: {e}")
    
    async def _handle_agent_message(self, message: Message):
        """Handle message from an agent."""
        if message.type == "task_accepted":
            task_id = message.content["task_id"]
            agent_id = message.sender
            self.workload_balancer.assign_task(task_id, agent_id)
            logger.debug(f"Task {task_id} accepted by agent {agent_id}")
        
        elif message.type == "task_completed":
            task_id = message.content["task_id"]
            result_data = message.content["result"]
            result = TaskResult(**result_data)
            
            self.completed_tasks[task_id] = result
            self.workload_balancer.complete_task(
                task_id, result.agent_id, True, result.execution_time
            )
            logger.info(f"Task {task_id} completed by agent {result.agent_id}")
        
        elif message.type == "task_failed":
            task_id = message.content["task_id"]
            error = message.content["error"]
            agent_id = message.sender
            
            self.workload_balancer.complete_task(task_id, agent_id, False, 0)
            
            # Check if we should retry
            self.retry_counts[task_id] += 1
            if task_id in self.failed_tasks:
                task = self.failed_tasks[task_id]
                if self.retry_counts[task_id] < task.max_retries:
                    await self.requeue_task(task)
                else:
                    logger.error(f"Task {task_id} failed after {task.max_retries} retries")
        
        elif message.type == "dom_conflict":
            task_id = message.content["task_id"]
            if task_id in self.failed_tasks:
                task = self.failed_tasks[task_id]
                await self.requeue_task(task)
    
    async def _assign_tasks(self):
        """Assign tasks to available agents."""
        while self.running:
            try:
                # Get next task from queue
                try:
                    priority, task = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Check dependencies
                if task.dependencies:
                    deps_completed = all(
                        dep_id in self.completed_tasks 
                        for dep_id in task.dependencies
                    )
                    if not deps_completed:
                        # Requeue with lower priority
                        await self.task_queue.put((priority + 1, task))
                        continue
                
                # Find best agent for task
                available_agents = list(self.agent_capabilities.values())
                best_agent_id = self.workload_balancer.get_best_agent(task, available_agents)
                
                if best_agent_id:
                    # Assign task to agent
                    await self.communication_bus.send_message(
                        self._orchestrator_id, best_agent_id,
                        "task_assignment", {"task": task.__dict__}
                    )
                    self.failed_tasks[task.task_id] = task
                    logger.debug(f"Assigned task {task.task_id} to agent {best_agent_id}")
                else:
                    # No suitable agent available, requeue
                    await self.task_queue.put((priority, task))
                    await asyncio.sleep(0.5)  # Backoff
                
            except Exception as e:
                logger.error(f"Task assignment error: {e}")
                await asyncio.sleep(1)
    
    async def requeue_task(self, task: TaskDefinition):
        """Requeue a failed task for retry."""
        await self.task_queue.put((task.priority.value, task))
        logger.info(f"Requeued task {task.task_id} for retry")
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task."""
        if task_id in self.completed_tasks:
            result = self.completed_tasks[task_id]
            return {
                "status": "completed",
                "result": result.__dict__,
                "agent_id": result.agent_id
            }
        elif task_id in self.failed_tasks:
            return {
                "status": "failed",
                "retry_count": self.retry_counts[task_id],
                "max_retries": self.failed_tasks[task_id].max_retries
            }
        else:
            # Check if in queue
            return {"status": "queued"}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            "agents": {
                agent_id: {
                    "type": cap.agent_type.name,
                    "workload": self.workload_balancer.agent_workloads.get(agent_id, 0),
                    "performance": self.workload_balancer.agent_performance.get(agent_id, 1.0)
                }
                for agent_id, cap in self.agent_capabilities.items()
            },
            "tasks": {
                "queued": self.task_queue.qsize(),
                "completed": len(self.completed_tasks),
                "failed": len(self.failed_tasks)
            },
            "resources": {
                "locks": len(self.resource_manager.locks),
                "dom_versions": len(self.resource_manager.dom_versions)
            }
        }


# Factory functions for easy agent creation
def create_navigation_agent(orchestrator: Orchestrator, **kwargs) -> NavigationAgent:
    """Create a navigation agent."""
    agent_id = f"nav_{str(uuid.uuid4())[:8]}"
    return NavigationAgent(
        agent_id=agent_id,
        orchestrator=orchestrator,
        **kwargs
    )


def create_form_filling_agent(orchestrator: Orchestrator, **kwargs) -> FormFillingAgent:
    """Create a form filling agent."""
    agent_id = f"form_{str(uuid.uuid4())[:8]}"
    return FormFillingAgent(
        agent_id=agent_id,
        orchestrator=orchestrator,
        **kwargs
    )


def create_data_extraction_agent(orchestrator: Orchestrator, **kwargs) -> DataExtractionAgent:
    """Create a data extraction agent."""
    agent_id = f"data_{str(uuid.uuid4())[:8]}"
    return DataExtractionAgent(
        agent_id=agent_id,
        orchestrator=orchestrator,
        **kwargs
    )


# Example usage
async def example_usage():
    """Example of how to use the orchestrator."""
    # Create orchestrator
    orchestrator = Orchestrator(max_agents=5)
    await orchestrator.start()
    
    # Create specialized agents
    nav_agent = create_navigation_agent(orchestrator)
    form_agent = create_form_filling_agent(orchestrator)
    data_agent = create_data_extraction_agent(orchestrator)
    
    # Register agents
    await orchestrator.register_agent(nav_agent, ["web_navigation"])
    await orchestrator.register_agent(form_agent, ["web_forms"])
    await orchestrator.register_agent(data_agent, ["web_scraping"])
    
    # Submit a complex task
    complex_task = {
        "type": "web_scraping",
        "url": "https://example.com",
        "selectors": {
            "titles": "h1",
            "paragraphs": "p"
        }
    }
    
    subtasks = await orchestrator.decompose_task(complex_task)
    await orchestrator.submit_tasks(subtasks)
    
    # Monitor progress
    for _ in range(30):  # Monitor for 30 seconds
        status = await orchestrator.get_system_status()
        print(f"System status: {json.dumps(status, indent=2)}")
        await asyncio.sleep(1)
    
    # Cleanup
    await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())