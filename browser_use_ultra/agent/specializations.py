"""
Parallel Agent Orchestration Engine for browser-use-ultra
Enables multiple AI agents to operate concurrently with intelligent resource sharing,
conflict resolution, and task decomposition.
"""

import asyncio
import uuid
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import logging
from contextlib import asynccontextmanager

from browser_use_ultra.agent.service import AgentService
from browser_use_ultra.agent.views import AgentState
from browser_use_ultra.actor.page import Page
from browser_use_ultra.actor.element import Element
from browser_use_ultra.agent.message_manager.service import MessageManager
from browser_use_ultra.agent.message_manager.views import Message
from browser_use_ultra.agent.prompts import AgentPrompts
from browser_use_ultra.agent.cloud_events import CloudEvent, CloudEventType

logger = logging.getLogger(__name__)


class AgentSpecialization(Enum):
    """Specialized agent types for parallel orchestration"""
    NAVIGATION = auto()
    FORM_FILLING = auto()
    DATA_EXTRACTION = auto()
    AUTHENTICATION = auto()
    MONITORING = auto()
    GENERAL = auto()


class TaskStatus(Enum):
    """Status of orchestrated tasks"""
    PENDING = auto()
    ASSIGNED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    BLOCKED = auto()


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving DOM conflicts"""
    OPTIMISTIC = auto()  # Last write wins with version checking
    PESSIMISTIC = auto()  # Lock-based with timeouts
    COLLABORATIVE = auto()  # Agents negotiate changes


@dataclass
class Task:
    """Represents a unit of work for an agent"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    description: str = ""
    required_specialization: AgentSpecialization = AgentSpecialization.GENERAL
    priority: int = 0  # Higher = more important
    dependencies: List[str] = field(default_factory=list)  # Task IDs
    data: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout: Optional[float] = None  # seconds


@dataclass
class AgentCapability:
    """Describes what an agent can do"""
    specialization: AgentSpecialization
    skills: Set[str] = field(default_factory=set)
    max_concurrent_tasks: int = 1
    supported_domains: Set[str] = field(default_factory=set)
    performance_score: float = 1.0  # Dynamic performance metric


@dataclass
class ResourceLock:
    """Manages access to shared resources"""
    resource_id: str
    holder_agent_id: Optional[str] = None
    lock_type: str = "exclusive"  # exclusive, shared
    acquired_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    version: int = 0  # For optimistic concurrency


@dataclass
class DOMVersion:
    """Tracks DOM state for optimistic concurrency control"""
    element_selector: str
    version_hash: str
    last_modified: datetime
    modified_by: Optional[str] = None


class AgentMessage:
    """Message passed between agents"""
    
    def __init__(self, sender_id: str, receiver_id: str, 
                 message_type: str, content: Any, 
                 correlation_id: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.message_type = message_type
        self.content = content
        self.correlation_id = correlation_id or self.id
        self.timestamp = datetime.now()
        self.handled = False


class SpecializedAgent:
    """Agent with specialization capabilities"""
    
    def __init__(self, agent_id: str, specialization: AgentSpecialization,
                 page: Page, capabilities: Optional[AgentCapability] = None):
        self.id = agent_id
        self.specialization = specialization
        self.page = page
        self.capabilities = capabilities or AgentCapability(specialization)
        self.state = AgentState.IDLE
        self.current_tasks: Dict[str, Task] = {}
        self.message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self.dom_versions: Dict[str, DOMVersion] = {}
        self.performance_history: List[float] = []
        
        # Initialize specialized components
        self.agent_service = AgentService(page=page)
        self.message_manager = MessageManager()
        self.prompts = AgentPrompts()
        
        # Specialization-specific configuration
        self._configure_for_specialization()
    
    def _configure_for_specialization(self):
        """Configure agent based on specialization"""
        if self.specialization == AgentSpecialization.NAVIGATION:
            self.capabilities.skills.update(["url_navigation", "link_clicking", "page_waiting"])
            self.capabilities.max_concurrent_tasks = 3
            
        elif self.specialization == AgentSpecialization.FORM_FILLING:
            self.capabilities.skills.update(["form_detection", "input_filling", "dropdown_selection"])
            self.capabilities.max_concurrent_tasks = 2
            
        elif self.specialization == AgentSpecialization.DATA_EXTRACTION:
            self.capabilities.skills.update(["element_scraping", "data_parsing", "pagination"])
            self.capabilities.max_concurrent_tasks = 5
            
        elif self.specialization == AgentSpecialization.AUTHENTICATION:
            self.capabilities.skills.update(["login_flow", "captcha_handling", "session_management"])
            self.capabilities.max_concurrent_tasks = 1
    
    async def process_message(self, message: AgentMessage):
        """Process incoming message from another agent"""
        try:
            if message.message_type == "task_request":
                # Handle task delegation request
                await self._handle_task_request(message)
            elif message.message_type == "resource_request":
                # Handle resource sharing request
                await self._handle_resource_request(message)
            elif message.message_type == "dom_conflict":
                # Handle DOM conflict notification
                await self._handle_dom_conflict(message)
            elif message.message_type == "status_update":
                # Handle status update from orchestrator
                await self._handle_status_update(message)
            
            message.handled = True
            
        except Exception as e:
            logger.error(f"Agent {self.id} failed to process message: {e}")
    
    async def _handle_task_request(self, message: AgentMessage):
        """Handle request to take on a task"""
        task_data = message.content.get("task")
        if task_data and self._can_handle_task(task_data):
            # Accept task
            response = AgentMessage(
                sender_id=self.id,
                receiver_id=message.sender_id,
                message_type="task_acceptance",
                content={"task_id": task_data["id"], "accepted": True},
                correlation_id=message.correlation_id
            )
            await self._send_message(response)
    
    async def _handle_resource_request(self, message: AgentMessage):
        """Handle request for shared resource access"""
        resource_id = message.content.get("resource_id")
        if resource_id:
            # Check if we can share the resource
            can_share = self._check_resource_sharing(resource_id)
            response = AgentMessage(
                sender_id=self.id,
                receiver_id=message.sender_id,
                message_type="resource_response",
                content={
                    "resource_id": resource_id,
                    "can_share": can_share,
                    "conditions": self._get_sharing_conditions(resource_id)
                },
                correlation_id=message.correlation_id
            )
            await self._send_message(response)
    
    async def _handle_dom_conflict(self, message: AgentMessage):
        """Handle DOM modification conflict"""
        conflict_data = message.content
        selector = conflict_data.get("selector")
        proposed_change = conflict_data.get("change")
        
        # Check current DOM version
        current_version = self.dom_versions.get(selector)
        if current_version:
            # Implement conflict resolution
            if conflict_data.get("strategy") == "optimistic":
                # Accept if version is newer
                if conflict_data.get("version", 0) > current_version.version:
                    self.dom_versions[selector] = DOMVersion(
                        element_selector=selector,
                        version_hash=conflict_data.get("new_hash"),
                        last_modified=datetime.now(),
                        modified_by=message.sender_id
                    )
    
    async def _handle_status_update(self, message: AgentMessage):
        """Handle status update from orchestrator"""
        status_data = message.content
        if "task_id" in status_data:
            task_id = status_data["task_id"]
            if task_id in self.current_tasks:
                # Update task status
                self.current_tasks[task_id].status = TaskStatus(status_data.get("status"))
    
    async def _send_message(self, message: AgentMessage):
        """Send message to another agent (via orchestrator)"""
        # Messages are routed through the orchestrator
        pass
    
    def _can_handle_task(self, task_data: Dict[str, Any]) -> bool:
        """Check if agent can handle the given task"""
        required_spec = task_data.get("required_specialization")
        if required_spec and required_spec != self.specialization:
            return False
        
        # Check if agent has required skills
        required_skills = set(task_data.get("required_skills", []))
        if not required_skills.issubset(self.capabilities.skills):
            return False
        
        # Check capacity
        if len(self.current_tasks) >= self.capabilities.max_concurrent_tasks:
            return False
        
        return True
    
    def _check_resource_sharing(self, resource_id: str) -> bool:
        """Check if resource can be shared"""
        # Implement resource sharing logic
        return True
    
    def _get_sharing_conditions(self, resource_id: str) -> Dict[str, Any]:
        """Get conditions for sharing a resource"""
        return {"timeout": 30, "read_only": False}
    
    async def execute_task(self, task: Task) -> Any:
        """Execute a task with specialization-specific logic"""
        self.state = AgentState.WORKING
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        try:
            # Route to specialized execution method
            if self.specialization == AgentSpecialization.NAVIGATION:
                result = await self._execute_navigation_task(task)
            elif self.specialization == AgentSpecialization.FORM_FILLING:
                result = await self._execute_form_filling_task(task)
            elif self.specialization == AgentSpecialization.DATA_EXTRACTION:
                result = await self._execute_data_extraction_task(task)
            elif self.specialization == AgentSpecialization.AUTHENTICATION:
                result = await self._execute_authentication_task(task)
            else:
                result = await self._execute_general_task(task)
            
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now()
            
            # Update performance metrics
            execution_time = (task.completed_at - task.started_at).total_seconds()
            self.performance_history.append(execution_time)
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            raise
        finally:
            self.state = AgentState.IDLE
            if task.id in self.current_tasks:
                del self.current_tasks[task.id]
    
    async def _execute_navigation_task(self, task: Task) -> Any:
        """Execute navigation-specific task"""
        url = task.data.get("url")
        if url:
            await self.page.goto(url)
            return {"navigated_to": url, "title": await self.page.title()}
        return {}
    
    async def _execute_form_filling_task(self, task: Task) -> Any:
        """Execute form-filling task"""
        form_data = task.data.get("form_data", {})
        results = {}
        
        for selector, value in form_data.items():
            element = await self.page.query_selector(selector)
            if element:
                await element.fill(value)
                results[selector] = "filled"
        
        return results
    
    async def _execute_data_extraction_task(self, task: Task) -> Any:
        """Execute data extraction task"""
        selectors = task.data.get("selectors", {})
        extracted_data = {}
        
        for name, selector in selectors.items():
            elements = await self.page.query_selector_all(selector)
            extracted_data[name] = [await el.text_content() for el in elements]
        
        return extracted_data
    
    async def _execute_authentication_task(self, task: Task) -> Any:
        """Execute authentication task"""
        credentials = task.data.get("credentials", {})
        login_url = task.data.get("login_url")
        
        if login_url:
            await self.page.goto(login_url)
        
        # Fill login form
        username_selector = task.data.get("username_selector", "#username")
        password_selector = task.data.get("password_selector", "#password")
        
        await self.page.fill(username_selector, credentials.get("username", ""))
        await self.page.fill(password_selector, credentials.get("password", ""))
        
        # Submit form
        submit_selector = task.data.get("submit_selector", "button[type='submit']")
        await self.page.click(submit_selector)
        
        return {"authenticated": True, "url": self.page.url}
    
    async def _execute_general_task(self, task: Task) -> Any:
        """Execute general task using agent service"""
        # Use the underlying agent service
        return await self.agent_service.execute_task(task.data)


class ResourceCoordinator:
    """Manages shared resources and conflict resolution"""
    
    def __init__(self, strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.OPTIMISTIC):
        self.strategy = strategy
        self.locks: Dict[str, ResourceLock] = {}
        self.dom_versions: Dict[str, DOMVersion] = {}
        self.lock_timeout = 30  # seconds
        
    async def acquire_lock(self, resource_id: str, agent_id: str, 
                          lock_type: str = "exclusive") -> bool:
        """Acquire a lock on a resource"""
        current_lock = self.locks.get(resource_id)
        
        if current_lock and current_lock.holder_agent_id:
            # Check if lock has expired
            if current_lock.expires_at and datetime.now() > current_lock.expires_at:
                # Lock expired, can acquire
                pass
            elif current_lock.holder_agent_id == agent_id:
                # Already own the lock
                return True
            else:
                # Lock held by another agent
                if lock_type == "shared" and current_lock.lock_type == "shared":
                    # Can share read lock
                    pass
                else:
                    return False
        
        # Acquire lock
        self.locks[resource_id] = ResourceLock(
            resource_id=resource_id,
            holder_agent_id=agent_id,
            lock_type=lock_type,
            acquired_at=datetime.now(),
            expires_at=datetime.now().timestamp() + self.lock_timeout,
            version=self.locks.get(resource_id, ResourceLock(resource_id)).version + 1
        )
        
        return True
    
    async def release_lock(self, resource_id: str, agent_id: str) -> bool:
        """Release a lock on a resource"""
        current_lock = self.locks.get(resource_id)
        
        if current_lock and current_lock.holder_agent_id == agent_id:
            current_lock.holder_agent_id = None
            return True
        
        return False
    
    def check_dom_version(self, selector: str, expected_version: int) -> bool:
        """Check if DOM element version matches expected"""
        dom_version = self.dom_versions.get(selector)
        if not dom_version:
            return True  # No version yet, can modify
        return dom_version.version_hash == str(expected_version)
    
    def update_dom_version(self, selector: str, agent_id: str, 
                          new_hash: str) -> None:
        """Update DOM element version"""
        self.dom_versions[selector] = DOMVersion(
            element_selector=selector,
            version_hash=new_hash,
            last_modified=datetime.now(),
            modified_by=agent_id
        )


class TaskDecomposer:
    """Decomposes complex tasks into parallelizable subtasks"""
    
    def __init__(self):
        self.decomposition_rules = {
            "multi_page_scrape": self._decompose_multi_page_scrape,
            "form_submission": self._decompose_form_submission,
            "authentication_flow": self._decompose_auth_flow,
            "data_pipeline": self._decompose_data_pipeline
        }
    
    async def decompose(self, task: Task) -> List[Task]:
        """Decompose a task into subtasks"""
        task_type = task.type
        
        if task_type in self.decomposition_rules:
            return await self.decomposition_rules[task_type](task)
        
        # Default: return task as-is
        return [task]
    
    async def _decompose_multi_page_scrape(self, task: Task) -> List[Task]:
        """Decompose multi-page scraping task"""
        urls = task.data.get("urls", [])
        subtasks = []
        
        for i, url in enumerate(urls):
            subtask = Task(
                id=f"{task.id}_page_{i}",
                type="single_page_scrape",
                description=f"Scrape page {i+1}",
                required_specialization=AgentSpecialization.DATA_EXTRACTION,
                priority=task.priority,
                data={"url": url, "selectors": task.data.get("selectors", {})}
            )
            subtasks.append(subtask)
        
        return subtasks
    
    async def _decompose_form_submission(self, task: Task) -> List[Task]:
        """Decompose form submission task"""
        form_sections = task.data.get("sections", [])
        subtasks = []
        
        for i, section in enumerate(form_sections):
            subtask = Task(
                id=f"{task.id}_section_{i}",
                type="form_section",
                description=f"Fill form section {i+1}",
                required_specialization=AgentSpecialization.FORM_FILLING,
                priority=task.priority,
                data={"section_data": section}
            )
            subtasks.append(subtask)
        
        return subtasks
    
    async def _decompose_auth_flow(self, task: Task) -> List[Task]:
        """Decompose authentication flow"""
        subtasks = []
        
        # Login task
        login_task = Task(
            id=f"{task.id}_login",
            type="login",
            description="Perform login",
            required_specialization=AgentSpecialization.AUTHENTICATION,
            priority=task.priority,
            data=task.data.get("login_data", {})
        )
        subtasks.append(login_task)
        
        # Post-login tasks (if any)
        post_login_tasks = task.data.get("post_login_tasks", [])
        for i, plt in enumerate(post_login_tasks):
            subtask = Task(
                id=f"{task.id}_post_login_{i}",
                type=plt.get("type", "general"),
                description=f"Post-login task {i+1}",
                dependencies=[login_task.id],
                data=plt
            )
            subtasks.append(subtask)
        
        return subtasks
    
    async def _decompose_data_pipeline(self, task: Task) -> List[Task]:
        """Decompose data pipeline task"""
        stages = task.data.get("stages", [])
        subtasks = []
        
        for i, stage in enumerate(stages):
            dependencies = []
            if i > 0:
                dependencies.append(f"{task.id}_stage_{i-1}")
            
            subtask = Task(
                id=f"{task.id}_stage_{i}",
                type=stage.get("type", "data_processing"),
                description=f"Pipeline stage {i+1}",
                required_specialization=AgentSpecialization.DATA_EXTRACTION,
                priority=task.priority,
                dependencies=dependencies,
                data=stage
            )
            subtasks.append(subtask)
        
        return subtasks


class WorkloadBalancer:
    """Balances workload across agents"""
    
    def __init__(self):
        self.agent_loads: Dict[str, int] = defaultdict(int)
        self.agent_performance: Dict[str, float] = defaultdict(lambda: 1.0)
        self.task_priorities: Dict[str, int] = {}
        
    def update_agent_load(self, agent_id: str, load: int):
        """Update agent's current load"""
        self.agent_loads[agent_id] = load
    
    def update_agent_performance(self, agent_id: str, performance: float):
        """Update agent's performance score"""
        self.agent_performance[agent_id] = performance
    
    def select_agent_for_task(self, task: Task, 
                             available_agents: List[SpecializedAgent]) -> Optional[SpecializedAgent]:
        """Select best agent for a task based on load balancing"""
        if not available_agents:
            return None
        
        # Filter by specialization
        specialized_agents = [
            a for a in available_agents 
            if a.specialization == task.required_specialization or 
               task.required_specialization == AgentSpecialization.GENERAL
        ]
        
        if not specialized_agents:
            return None
        
        # Calculate scores for each agent
        agent_scores = []
        for agent in specialized_agents:
            load_score = 1.0 / (self.agent_loads[agent.id] + 1)
            performance_score = self.agent_performance[agent.id]
            skill_match = len(agent.capabilities.skills.intersection(
                set(task.data.get("required_skills", []))
            ))
            
            total_score = (load_score * 0.4 + 
                          performance_score * 0.4 + 
                          skill_match * 0.2)
            
            agent_scores.append((agent, total_score))
        
        # Select agent with highest score
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        return agent_scores[0][0] if agent_scores else None
    
    def balance_loads(self, agents: List[SpecializedAgent]) -> Dict[str, int]:
        """Calculate balanced load distribution"""
        total_load = sum(self.agent_loads.values())
        avg_load = total_load / len(agents) if agents else 0
        
        balance_plan = {}
        for agent in agents:
            current_load = self.agent_loads[agent.id]
            if current_load > avg_load * 1.5:  # Overloaded
                balance_plan[agent.id] = -1  # Reduce load
            elif current_load < avg_load * 0.5:  # Underloaded
                balance_plan[agent.id] = 1  # Increase load
            else:
                balance_plan[agent.id] = 0  # Balanced
        
        return balance_plan


class ParallelOrchestrator:
    """Central orchestrator for parallel agent execution"""
    
    def __init__(self, max_agents: int = 10, 
                 conflict_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.OPTIMISTIC):
        self.max_agents = max_agents
        self.agents: Dict[str, SpecializedAgent] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}
        
        # Core components
        self.resource_coordinator = ResourceCoordinator(conflict_strategy)
        self.task_decomposer = TaskDecomposer()
        self.workload_balancer = WorkloadBalancer()
        
        # Communication
        self.message_broker: Dict[str, asyncio.Queue[AgentMessage]] = {}
        
        # State
        self.running = False
        self.orchestration_task: Optional[asyncio.Task] = None
        self.agent_tasks: Dict[str, asyncio.Task] = {}
        
        # Event handlers
        self.event_handlers: Dict[CloudEventType, List[Callable]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            "tasks_processed": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "average_completion_time": 0,
            "agent_utilization": 0.0
        }
    
    async def start(self):
        """Start the orchestrator"""
        if self.running:
            return
        
        self.running = True
        self.orchestration_task = asyncio.create_task(self._orchestration_loop())
        logger.info("Parallel orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator"""
        self.running = False
        
        if self.orchestration_task:
            self.orchestration_task.cancel()
            try:
                await self.orchestration_task
            except asyncio.CancelledError:
                pass
        
        # Stop all agents
        for agent_id, task in self.agent_tasks.items():
            task.cancel()
        
        # Wait for agents to stop
        if self.agent_tasks:
            await asyncio.gather(*self.agent_tasks.values(), return_exceptions=True)
        
        logger.info("Parallel orchestrator stopped")
    
    async def register_agent(self, agent: SpecializedAgent) -> str:
        """Register an agent with the orchestrator"""
        if len(self.agents) >= self.max_agents:
            raise ValueError(f"Maximum agent limit ({self.max_agents}) reached")
        
        self.agents[agent.id] = agent
        self.message_broker[agent.id] = asyncio.Queue()
        self.workload_balancer.update_agent_load(agent.id, 0)
        
        # Start agent message processing
        self.agent_tasks[agent.id] = asyncio.create_task(
            self._process_agent_messages(agent.id)
        )
        
        logger.info(f"Agent {agent.id} registered with specialization {agent.specialization.name}")
        return agent.id
    
    async def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            # Cancel agent task
            if agent_id in self.agent_tasks:
                self.agent_tasks[agent_id].cancel()
                try:
                    await self.agent_tasks[agent_id]
                except asyncio.CancelledError:
                    pass
                del self.agent_tasks[agent_id]
            
            # Clean up
            del self.agents[agent_id]
            del self.message_broker[agent_id]
            self.workload_balancer.update_agent_load(agent_id, 0)
            
            logger.info(f"Agent {agent_id} unregistered")
    
    async def submit_task(self, task: Task) -> str:
        """Submit a task for execution"""
        # Decompose task if needed
        subtasks = await self.task_decomposer.decompose(task)
        
        # Add subtasks to queue
        for subtask in subtasks:
            # Priority queue uses negative priority for max-heap behavior
            await self.task_queue.put((-subtask.priority, subtask.created_at, subtask))
        
        logger.info(f"Task {task.id} submitted with {len(subtasks)} subtasks")
        return task.id
    
    async def submit_tasks(self, tasks: List[Task]) -> List[str]:
        """Submit multiple tasks"""
        task_ids = []
        for task in tasks:
            task_id = await self.submit_task(task)
            task_ids.append(task_id)
        return task_ids
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a task"""
        # Check completed tasks
        if task_id in self.completed_tasks:
            return TaskStatus.COMPLETED
        
        # Check failed tasks
        if task_id in self.failed_tasks:
            return TaskStatus.FAILED
        
        # Check agents for in-progress tasks
        for agent in self.agents.values():
            if task_id in agent.current_tasks:
                return agent.current_tasks[task_id].status
        
        # Check queue
        # Note: This is simplified; in production you'd need to track queued tasks
        return TaskStatus.PENDING
    
    async def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get result of a completed task"""
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].result
        return None
    
    async def send_message_to_agent(self, sender_id: str, receiver_id: str,
                                   message_type: str, content: Any) -> bool:
        """Send a message from one agent to another"""
        if receiver_id not in self.message_broker:
            return False
        
        message = AgentMessage(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content
        )
        
        await self.message_broker[receiver_id].put(message)
        return True
    
    async def broadcast_message(self, sender_id: str, message_type: str,
                               content: Any, exclude: Optional[Set[str]] = None):
        """Broadcast a message to all agents"""
        exclude = exclude or set()
        
        for agent_id in self.message_broker:
            if agent_id != sender_id and agent_id not in exclude:
                await self.send_message_to_agent(
                    sender_id, agent_id, message_type, content
                )
    
    async def acquire_resource_lock(self, agent_id: str, resource_id: str,
                                   lock_type: str = "exclusive") -> bool:
        """Acquire a resource lock for an agent"""
        return await self.resource_coordinator.acquire_lock(
            resource_id, agent_id, lock_type
        )
    
    async def release_resource_lock(self, agent_id: str, resource_id: str) -> bool:
        """Release a resource lock"""
        return await self.resource_coordinator.release_lock(resource_id, agent_id)
    
    async def _orchestration_loop(self):
        """Main orchestration loop"""
        while self.running:
            try:
                # Get next task from queue
                try:
                    priority, created_at, task = await asyncio.wait_for(
                        self.task_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # No tasks, check for load balancing
                    await self._balance_workload()
                    continue
                
                # Check if task dependencies are met
                if not await self._check_dependencies(task):
                    # Requeue task with lower priority
                    await self.task_queue.put((priority + 1, created_at, task))
                    continue
                
                # Find available agent
                available_agents = [
                    a for a in self.agents.values()
                    if a.state == AgentState.IDLE and
                       len(a.current_tasks) < a.capabilities.max_concurrent_tasks
                ]
                
                selected_agent = self.workload_balancer.select_agent_for_task(
                    task, available_agents
                )
                
                if selected_agent:
                    # Assign task to agent
                    await self._assign_task_to_agent(task, selected_agent)
                else:
                    # No agent available, requeue
                    await self.task_queue.put((priority, created_at, task))
                    await asyncio.sleep(0.1)  # Prevent busy waiting
                
                # Update statistics
                self.stats["tasks_processed"] += 1
                
            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}")
                await asyncio.sleep(1)
    
    async def _check_dependencies(self, task: Task) -> bool:
        """Check if task dependencies are satisfied"""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
    
    async def _assign_task_to_agent(self, task: Task, agent: SpecializedAgent):
        """Assign a task to an agent"""
        task.status = TaskStatus.ASSIGNED
        task.assigned_agent = agent.id
        
        # Add to agent's current tasks
        agent.current_tasks[task.id] = task
        
        # Update workload balancer
        self.workload_balancer.update_agent_load(
            agent.id, len(agent.current_tasks)
        )
        
        # Create task execution coroutine
        asyncio.create_task(self._execute_agent_task(agent, task))
        
        logger.info(f"Task {task.id} assigned to agent {agent.id}")
    
    async def _execute_agent_task(self, agent: SpecializedAgent, task: Task):
        """Execute a task on an agent"""
        try:
            result = await agent.execute_task(task)
            
            # Store result
            self.completed_tasks[task.id] = task
            self.stats["tasks_succeeded"] += 1
            
            # Update agent performance
            if task.started_at and task.completed_at:
                execution_time = (task.completed_at - task.started_at).total_seconds()
                # Update performance score (lower time = better performance)
                new_performance = 1.0 / max(execution_time, 0.1)
                self.workload_balancer.update_agent_performance(
                    agent.id, new_performance
                )
            
            # Notify completion
            await self._notify_task_completion(task)
            
        except Exception as e:
            logger.error(f"Task {task.id} failed on agent {agent.id}: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self.failed_tasks[task.id] = task
            self.stats["tasks_failed"] += 1
            
            # Notify failure
            await self._notify_task_failure(task, str(e))
        
        finally:
            # Remove from agent's current tasks
            if task.id in agent.current_tasks:
                del agent.current_tasks[task.id]
            
            # Update workload
            self.workload_balancer.update_agent_load(
                agent.id, len(agent.current_tasks)
            )
    
    async def _notify_task_completion(self, task: Task):
        """Notify about task completion"""
        event = CloudEvent(
            type=CloudEventType.TASK_COMPLETED,
            data={
                "task_id": task.id,
                "result": task.result,
                "execution_time": (task.completed_at - task.started_at).total_seconds() 
                    if task.completed_at and task.started_at else None
            }
        )
        
        # Notify all handlers
        for handler in self.event_handlers.get(CloudEventType.TASK_COMPLETED, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
    
    async def _notify_task_failure(self, task: Task, error: str):
        """Notify about task failure"""
        event = CloudEvent(
            type=CloudEventType.TASK_FAILED,
            data={
                "task_id": task.id,
                "error": error,
                "assigned_agent": task.assigned_agent
            }
        )
        
        for handler in self.event_handlers.get(CloudEventType.TASK_FAILED, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
    
    async def _balance_workload(self):
        """Balance workload across agents"""
        balance_plan = self.workload_balancer.balance_loads(list(self.agents.values()))
        
        for agent_id, adjustment in balance_plan.items():
            if adjustment < 0:  # Agent is overloaded
                # Could implement task migration here
                pass
            elif adjustment > 0:  # Agent is underloaded
                # Agent can take more tasks
                pass
    
    async def _process_agent_messages(self, agent_id: str):
        """Process messages for an agent"""
        message_queue = self.message_broker.get(agent_id)
        if not message_queue:
            return
        
        while self.running:
            try:
                message = await message_queue.get()
                agent = self.agents.get(agent_id)
                
                if agent:
                    await agent.process_message(message)
                
                message_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing message for agent {agent_id}: {e}")
    
    def register_event_handler(self, event_type: CloudEventType, 
                              handler: Callable[[CloudEvent], Awaitable[None]]):
        """Register an event handler"""
        self.event_handlers[event_type].append(handler)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        agent_utilization = 0.0
        if self.agents:
            busy_agents = sum(1 for a in self.agents.values() 
                            if a.state == AgentState.WORKING)
            agent_utilization = busy_agents / len(self.agents)
        
        self.stats["agent_utilization"] = agent_utilization
        
        return {
            **self.stats,
            "active_agents": len(self.agents),
            "queued_tasks": self.task_queue.qsize(),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks)
        }
    
    @asynccontextmanager
    async def orchestration_session(self):
        """Context manager for orchestration session"""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()


# Factory functions for creating specialized agents
def create_navigation_agent(page: Page, agent_id: Optional[str] = None) -> SpecializedAgent:
    """Create a navigation-specialized agent"""
    agent_id = agent_id or f"nav_agent_{uuid.uuid4().hex[:8]}"
    return SpecializedAgent(
        agent_id=agent_id,
        specialization=AgentSpecialization.NAVIGATION,
        page=page
    )


def create_form_filling_agent(page: Page, agent_id: Optional[str] = None) -> SpecializedAgent:
    """Create a form-filling-specialized agent"""
    agent_id = agent_id or f"form_agent_{uuid.uuid4().hex[:8]}"
    return SpecializedAgent(
        agent_id=agent_id,
        specialization=AgentSpecialization.FORM_FILLING,
        page=page
    )


def create_data_extraction_agent(page: Page, agent_id: Optional[str] = None) -> SpecializedAgent:
    """Create a data-extraction-specialized agent"""
    agent_id = agent_id or f"data_agent_{uuid.uuid4().hex[:8]}"
    return SpecializedAgent(
        agent_id=agent_id,
        specialization=AgentSpecialization.DATA_EXTRACTION,
        page=page
    )


def create_authentication_agent(page: Page, agent_id: Optional[str] = None) -> SpecializedAgent:
    """Create an authentication-specialized agent"""
    agent_id = agent_id or f"auth_agent_{uuid.uuid4().hex[:8]}"
    return SpecializedAgent(
        agent_id=agent_id,
        specialization=AgentSpecialization.AUTHENTICATION,
        page=page
    )


# Utility functions for common orchestration patterns
async def parallel_scrape(urls: List[str], selectors: Dict[str, str],
                         orchestrator: ParallelOrchestrator,
                         max_concurrent: int = 5) -> Dict[str, Any]:
    """Scrape multiple URLs in parallel"""
    # Create data extraction agents
    agents = []
    for i in range(min(max_concurrent, len(urls))):
        # In real implementation, you'd create actual pages
        # For now, we'll create mock agents
        agent = create_data_extraction_agent(page=None, agent_id=f"scraper_{i}")
        await orchestrator.register_agent(agent)
        agents.append(agent)
    
    # Create scraping tasks
    tasks = []
    for i, url in enumerate(urls):
        task = Task(
            id=f"scrape_{i}",
            type="single_page_scrape",
            description=f"Scrape {url}",
            required_specialization=AgentSpecialization.DATA_EXTRACTION,
            data={"url": url, "selectors": selectors}
        )
        tasks.append(task)
    
    # Submit tasks
    task_ids = await orchestrator.submit_tasks(tasks)
    
    # Wait for completion (simplified)
    results = {}
    for task_id, url in zip(task_ids, urls):
        # In real implementation, you'd poll for completion
        # For now, we'll just return the task IDs
        results[url] = {"task_id": task_id, "status": "submitted"}
    
    return results


async def orchestrated_login_flow(login_url: str, credentials: Dict[str, str],
                                 post_login_tasks: Optional[List[Dict[str, Any]]] = None,
                                 orchestrator: Optional[ParallelOrchestrator] = None) -> Dict[str, Any]:
    """Execute a login flow with optional post-login tasks"""
    if orchestrator is None:
        orchestrator = ParallelOrchestrator()
        await orchestrator.start()
    
    # Create authentication agent
    auth_agent = create_authentication_agent(page=None)
    await orchestrator.register_agent(auth_agent)
    
    # Create login task
    login_task = Task(
        id=f"login_{uuid.uuid4().hex[:8]}",
        type="authentication_flow",
        description="Perform login and post-login tasks",
        required_specialization=AgentSpecialization.AUTHENTICATION,
        data={
            "login_url": login_url,
            "credentials": credentials,
            "post_login_tasks": post_login_tasks or []
        }
    )
    
    # Submit task
    task_id = await orchestrator.submit_task(login_task)
    
    return {
        "task_id": task_id,
        "agent_id": auth_agent.id,
        "status": "submitted"
    }