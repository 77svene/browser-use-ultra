"""Parallel Agent Orchestration Engine for concurrent browser automation with intelligent resource management."""

import asyncio
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from browser_use_ultra.actor.page import Page
from browser_use_ultra.agent.service import AgentService
from browser_use_ultra.agent.views import AgentState

logger = logging.getLogger(__name__)


class AgentSpecialization(Enum):
    """Specialized agent types for different browser automation tasks."""
    NAVIGATION = "navigation"
    FORM_FILLING = "form_filling"
    DATA_EXTRACTION = "data_extraction"
    INTERACTION = "interaction"
    MONITORING = "monitoring"
    GENERAL = "general"


class TaskPriority(Enum):
    """Priority levels for task scheduling."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class Task:
    """Represents a unit of work for an agent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    specialization: AgentSpecialization = AgentSpecialization.GENERAL
    priority: TaskPriority = TaskPriority.MEDIUM
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[Exception] = None
    created_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 30.0
    page_url: Optional[str] = None
    requires_login: bool = False
    shared_resources: List[str] = field(default_factory=list)


@dataclass
class AgentInfo:
    """Information about an agent in the pool."""
    id: str
    specialization: AgentSpecialization
    capabilities: Set[str] = field(default_factory=set)
    current_task: Optional[str] = None
    status: str = "idle"
    performance_score: float = 1.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    last_active: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    page_context: Optional[str] = None
    login_state: Optional[Dict[str, Any]] = None
    resource_locks: Set[str] = field(default_factory=set)


class ResourceManager:
    """Manages shared resources like login states, cookies, and page contexts."""
    
    def __init__(self):
        self.login_states: Dict[str, Dict[str, Any]] = {}
        self.cookies: Dict[str, List[Dict[str, Any]]] = {}
        self.page_contexts: Dict[str, Dict[str, Any]] = {}
        self.locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.resource_usage: Dict[str, Set[str]] = defaultdict(set)  # resource -> agent_ids
        
    async def acquire_lock(self, resource_id: str, agent_id: str, timeout: float = 10.0) -> bool:
        """Acquire a lock on a shared resource with timeout."""
        try:
            await asyncio.wait_for(self.locks[resource_id].acquire(), timeout=timeout)
            self.resource_usage[resource_id].add(agent_id)
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Agent {agent_id} failed to acquire lock on {resource_id}")
            return False
    
    def release_lock(self, resource_id: str, agent_id: str):
        """Release a lock on a shared resource."""
        if resource_id in self.locks:
            self.locks[resource_id].release()
            self.resource_usage[resource_id].discard(agent_id)
    
    def get_login_state(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get login state for a domain."""
        return self.login_states.get(domain)
    
    def set_login_state(self, domain: str, state: Dict[str, Any], agent_id: str):
        """Set login state for a domain."""
        self.login_states[domain] = state
        logger.info(f"Agent {agent_id} updated login state for {domain}")
    
    def get_page_context(self, url: str) -> Optional[Dict[str, Any]]:
        """Get shared context for a page."""
        return self.page_contexts.get(url)
    
    def update_page_context(self, url: str, context: Dict[str, Any], agent_id: str):
        """Update shared context for a page."""
        if url not in self.page_contexts:
            self.page_contexts[url] = {}
        self.page_contexts[url].update(context)
        logger.debug(f"Agent {agent_id} updated context for {url}")
    
    def get_resource_conflicts(self) -> List[Tuple[str, Set[str]]]:
        """Get resources with multiple agents accessing them."""
        return [(res, agents) for res, agents in self.resource_usage.items() if len(agents) > 1]


class ConflictResolver:
    """Handles conflicts between concurrent agent operations."""
    
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.dom_version: Dict[str, int] = defaultdict(int)  # page_url -> version
        self.pending_modifications: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.conflict_handlers: Dict[str, Callable] = {}
        
    async def check_dom_conflict(self, page_url: str, expected_version: int) -> bool:
        """Check if DOM has been modified since expected version."""
        current_version = self.dom_version.get(page_url, 0)
        return current_version != expected_version
    
    async def register_dom_modification(self, page_url: str, modification: Dict[str, Any], agent_id: str):
        """Register a DOM modification and increment version."""
        self.dom_version[page_url] = self.dom_version.get(page_url, 0) + 1
        self.pending_modifications[page_url].append({
            "agent_id": agent_id,
            "modification": modification,
            "version": self.dom_version[page_url],
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Keep only recent modifications (last 100)
        if len(self.pending_modifications[page_url]) > 100:
            self.pending_modifications[page_url] = self.pending_modifications[page_url][-100:]
    
    async def resolve_conflict(self, page_url: str, agent_id: str, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict for an operation on a page."""
        # Simple strategy: retry with exponential backoff
        retry_count = operation.get("retry_count", 0)
        if retry_count < 3:
            wait_time = 0.1 * (2 ** retry_count)  # Exponential backoff
            await asyncio.sleep(wait_time)
            operation["retry_count"] = retry_count + 1
            return {"action": "retry", "wait_time": wait_time}
        else:
            # If too many retries, try to merge or skip
            logger.warning(f"Agent {agent_id} hit max retries for operation on {page_url}")
            return {"action": "skip", "reason": "max_retries_exceeded"}


class TaskDecomposer:
    """Decomposes complex tasks into specialized subtasks."""
    
    def __init__(self):
        self.decomposition_rules: Dict[str, Callable] = {
            "fill_form": self._decompose_form_filling,
            "extract_data": self._decompose_data_extraction,
            "navigate_and_extract": self._decompose_navigation_extraction,
            "multi_step_workflow": self._decompose_workflow
        }
    
    def decompose(self, task: Task) -> List[Task]:
        """Decompose a task into subtasks based on its description and parameters."""
        task_type = task.parameters.get("type", "")
        
        if task_type in self.decomposition_rules:
            return self.decomposition_rules[task_type](task)
        
        # Default: return the task as-is
        return [task]
    
    def _decompose_form_filling(self, task: Task) -> List[Task]:
        """Decompose form filling into navigation, field detection, and filling."""
        subtasks = []
        form_url = task.parameters.get("form_url")
        form_fields = task.parameters.get("fields", {})
        
        if form_url:
            # Navigation task
            nav_task = Task(
                description=f"Navigate to form at {form_url}",
                specialization=AgentSpecialization.NAVIGATION,
                priority=task.priority,
                parameters={"url": form_url},
                page_url=form_url
            )
            subtasks.append(nav_task)
        
        # Form filling task
        fill_task = Task(
            description=f"Fill form fields",
            specialization=AgentSpecialization.FORM_FILLING,
            priority=task.priority,
            parameters={"fields": form_fields},
            dependencies=[nav_task.id] if form_url else [],
            page_url=form_url,
            requires_login=task.requires_login
        )
        subtasks.append(fill_task)
        
        return subtasks
    
    def _decompose_data_extraction(self, task: Task) -> List[Task]:
        """Decompose data extraction into navigation and extraction tasks."""
        subtasks = []
        urls = task.parameters.get("urls", [])
        
        for i, url in enumerate(urls):
            # Navigation task
            nav_task = Task(
                description=f"Navigate to {url}",
                specialization=AgentSpecialization.NAVIGATION,
                priority=task.priority,
                parameters={"url": url},
                page_url=url
            )
            
            # Extraction task
            extract_task = Task(
                description=f"Extract data from {url}",
                specialization=AgentSpecialization.DATA_EXTRACTION,
                priority=task.priority,
                parameters={"selectors": task.parameters.get("selectors", {})},
                dependencies=[nav_task.id],
                page_url=url
            )
            
            subtasks.extend([nav_task, extract_task])
        
        return subtasks
    
    def _decompose_navigation_extraction(self, task: Task) -> List[Task]:
        """Decompose navigation and extraction workflow."""
        url = task.parameters.get("url")
        selectors = task.parameters.get("selectors", {})
        
        nav_task = Task(
            description=f"Navigate to {url}",
            specialization=AgentSpecialization.NAVIGATION,
            priority=task.priority,
            parameters={"url": url},
            page_url=url
        )
        
        extract_task = Task(
            description=f"Extract data from {url}",
            specialization=AgentSpecialization.DATA_EXTRACTION,
            priority=task.priority,
            parameters={"selectors": selectors},
            dependencies=[nav_task.id],
            page_url=url
        )
        
        return [nav_task, extract_task]
    
    def _decompose_workflow(self, task: Task) -> List[Task]:
        """Decompose a multi-step workflow."""
        steps = task.parameters.get("steps", [])
        subtasks = []
        previous_task_id = None
        
        for i, step in enumerate(steps):
            step_task = Task(
                description=step.get("description", f"Step {i+1}"),
                specialization=AgentSpecialization(step.get("specialization", "general")),
                priority=task.priority,
                parameters=step.get("parameters", {}),
                dependencies=[previous_task_id] if previous_task_id else [],
                page_url=step.get("url"),
                requires_login=step.get("requires_login", False)
            )
            subtasks.append(step_task)
            previous_task_id = step_task.id
        
        return subtasks


class WorkloadBalancer:
    """Balances workload across agents based on capabilities and performance."""
    
    def __init__(self):
        self.agent_scores: Dict[str, float] = {}
        self.task_requirements: Dict[str, Set[str]] = {}
        
    def update_agent_score(self, agent_id: str, score_delta: float):
        """Update agent performance score."""
        if agent_id not in self.agent_scores:
            self.agent_scores[agent_id] = 1.0
        self.agent_scores[agent_id] = max(0.1, self.agent_scores[agent_id] + score_delta)
    
    def calculate_agent_fitness(self, agent: AgentInfo, task: Task) -> float:
        """Calculate how well an agent fits a task."""
        fitness = 0.0
        
        # Specialization match
        if agent.specialization == task.specialization:
            fitness += 3.0
        elif agent.specialization == AgentSpecialization.GENERAL:
            fitness += 1.0
        
        # Performance score
        fitness += self.agent_scores.get(agent.id, 1.0) * 2.0
        
        # Availability (prefer idle agents)
        if agent.status == "idle":
            fitness += 2.0
        elif agent.status == "busy":
            fitness -= 1.0
        
        # Resource compatibility
        if task.requires_login and agent.login_state:
            fitness += 1.5
        
        # Page context match
        if task.page_url and agent.page_context and agent.page_context.get("url") == task.page_url:
            fitness += 1.0
        
        return fitness
    
    def select_agent(self, agents: List[AgentInfo], task: Task) -> Optional[AgentInfo]:
        """Select the best agent for a task."""
        available_agents = [a for a in agents if a.status in ["idle", "busy"]]
        
        if not available_agents:
            return None
        
        # Calculate fitness for each agent
        agent_fitness = []
        for agent in available_agents:
            fitness = self.calculate_agent_fitness(agent, task)
            agent_fitness.append((agent, fitness))
        
        # Sort by fitness (descending)
        agent_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Return the best agent
        return agent_fitness[0][0] if agent_fitness else None
    
    def balance_load(self, agents: List[AgentInfo], tasks: List[Task]) -> Dict[str, List[str]]:
        """Balance load across agents, returning agent_id -> task_ids mapping."""
        assignments = defaultdict(list)
        unassigned_tasks = tasks.copy()
        
        # Sort tasks by priority
        unassigned_tasks.sort(key=lambda t: t.priority.value)
        
        for task in unassigned_tasks:
            best_agent = self.select_agent(agents, task)
            if best_agent:
                assignments[best_agent.id].append(task.id)
                # Update agent status
                best_agent.status = "busy"
                best_agent.current_task = task.id
        
        return assignments


class AgentOrchestrator:
    """Central orchestrator for parallel agent execution with intelligent resource management."""
    
    def __init__(self, max_agents: int = 5, shared_browser_context: bool = True):
        self.agent_pool: Dict[str, AgentInfo] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}
        self.resource_manager = ResourceManager()
        self.conflict_resolver = ConflictResolver(self.resource_manager)
        self.task_decomposer = TaskDecomposer()
        self.workload_balancer = WorkloadBalancer()
        self.max_agents = max_agents
        self.shared_browser_context = shared_browser_context
        self.agent_services: Dict[str, AgentService] = {}
        self.running = False
        self.orchestrator_task: Optional[asyncio.Task] = None
        self.agent_tasks: Dict[str, asyncio.Task] = {}
        self.message_queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "avg_task_time": 0.0
        }
        
        logger.info(f"AgentOrchestrator initialized with max_agents={max_agents}")
    
    async def start(self):
        """Start the orchestrator."""
        if self.running:
            return
        
        self.running = True
        self.orchestrator_task = asyncio.create_task(self._orchestration_loop())
        logger.info("AgentOrchestrator started")
    
    async def stop(self):
        """Stop the orchestrator and all agents."""
        self.running = False
        
        # Cancel orchestrator task
        if self.orchestrator_task:
            self.orchestrator_task.cancel()
            try:
                await self.orchestrator_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all agent tasks
        for task in self.agent_tasks.values():
            task.cancel()
        
        if self.agent_tasks:
            await asyncio.gather(*self.agent_tasks.values(), return_exceptions=True)
        
        # Clean up agent services
        for service in self.agent_services.values():
            if hasattr(service, 'close'):
                await service.close()
        
        logger.info("AgentOrchestrator stopped")
    
    async def submit_task(self, task: Task) -> str:
        """Submit a task to the orchestrator."""
        # Decompose task if needed
        subtasks = self.task_decomposer.decompose(task)
        
        for subtask in subtasks:
            # Add to priority queue (lower priority value = higher priority)
            await self.task_queue.put((subtask.priority.value, subtask))
            self.stats["tasks_submitted"] += 1
            logger.debug(f"Task submitted: {subtask.id} - {subtask.description}")
        
        return task.id
    
    async def _orchestration_loop(self):
        """Main orchestration loop."""
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
                
                # Check if dependencies are satisfied
                if not await self._check_dependencies(task):
                    # Re-queue task with lower priority
                    await self.task_queue.put((priority + 1, task))
                    continue
                
                # Find available agent or create new one
                agent_id = await self._assign_task(task)
                
                if agent_id:
                    # Execute task on agent
                    self.agent_tasks[task.id] = asyncio.create_task(
                        self._execute_task(agent_id, task)
                    )
                else:
                    # No agent available, re-queue with higher priority
                    await self.task_queue.put((max(0, priority - 1), task))
                    await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
                # Update workload balancing
                await self._update_workload_balancing()
                
                # Handle conflicts
                await self._handle_conflicts()
                
            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _check_dependencies(self, task: Task) -> bool:
        """Check if all task dependencies are satisfied."""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
    
    async def _assign_task(self, task: Task) -> Optional[str]:
        """Assign a task to an agent."""
        # Get list of agents
        agents = list(self.agent_pool.values())
        
        # Use workload balancer to select best agent
        best_agent = self.workload_balancer.select_agent(agents, task)
        
        if best_agent:
            task.assigned_agent = best_agent.id
            best_agent.current_task = task.id
            best_agent.status = "busy"
            return best_agent.id
        
        # Try to create new agent if under limit
        if len(self.agent_pool) < self.max_agents:
            agent_id = await self._create_agent(task.specialization)
            if agent_id:
                task.assigned_agent = agent_id
                self.agent_pool[agent_id].current_task = task.id
                self.agent_pool[agent_id].status = "busy"
                return agent_id
        
        return None
    
    async def _create_agent(self, specialization: AgentSpecialization) -> Optional[str]:
        """Create a new agent with specified specialization."""
        agent_id = f"agent_{len(self.agent_pool)}_{specialization.value}"
        
        try:
            # Create agent service
            agent_service = AgentService(
                agent_id=agent_id,
                specialization=specialization.value
            )
            
            # Create agent info
            agent_info = AgentInfo(
                id=agent_id,
                specialization=specialization,
                capabilities=self._get_capabilities_for_specialization(specialization)
            )
            
            # Store in pool
            self.agent_pool[agent_id] = agent_info
            self.agent_services[agent_id] = agent_service
            
            logger.info(f"Created new agent: {agent_id} with specialization {specialization.value}")
            return agent_id
            
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            return None
    
    def _get_capabilities_for_specialization(self, specialization: AgentSpecialization) -> Set[str]:
        """Get capabilities for a specialization."""
        capabilities_map = {
            AgentSpecialization.NAVIGATION: {"page_navigation", "url_handling", "wait_for_load"},
            AgentSpecialization.FORM_FILLING: {"form_detection", "field_filling", "form_submission"},
            AgentSpecialization.DATA_EXTRACTION: {"element_selection", "text_extraction", "data_parsing"},
            AgentSpecialization.INTERACTION: {"click", "hover", "scroll", "keyboard_input"},
            AgentSpecialization.MONITORING: {"element_monitoring", "state_tracking", "event_handling"},
            AgentSpecialization.GENERAL: {"navigation", "interaction", "extraction", "form_filling"}
        }
        return capabilities_map.get(specialization, set())
    
    async def _execute_task(self, agent_id: str, task: Task):
        """Execute a task on an agent."""
        agent_info = self.agent_pool.get(agent_id)
        agent_service = self.agent_services.get(agent_id)
        
        if not agent_info or not agent_service:
            logger.error(f"Agent {agent_id} not found")
            return
        
        task.status = "running"
        task.started_at = asyncio.get_event_loop().time()
        
        try:
            # Acquire necessary resource locks
            for resource in task.shared_resources:
                if not await self.resource_manager.acquire_lock(resource, agent_id, timeout=task.timeout):
                    raise TimeoutError(f"Failed to acquire lock on {resource}")
            
            # Check for login state if required
            if task.requires_login and task.page_url:
                from urllib.parse import urlparse
                domain = urlparse(task.page_url).netloc
                login_state = self.resource_manager.get_login_state(domain)
                if login_state:
                    agent_info.login_state = login_state
                    logger.debug(f"Agent {agent_id} using shared login state for {domain}")
            
            # Execute the task using agent service
            result = await agent_service.execute_task(
                task_description=task.description,
                parameters=task.parameters,
                timeout=task.timeout
            )
            
            # Update task with result
            task.result = result
            task.status = "completed"
            task.completed_at = asyncio.get_event_loop().time()
            
            # Update agent performance
            execution_time = task.completed_at - task.started_at
            self.workload_balancer.update_agent_score(agent_id, 0.1)  # Positive score for success
            agent_info.tasks_completed += 1
            
            # Store completed task
            self.completed_tasks[task.id] = task
            
            # Update statistics
            self.stats["tasks_completed"] += 1
            total_time = self.stats["avg_task_time"] * (self.stats["tasks_completed"] - 1) + execution_time
            self.stats["avg_task_time"] = total_time / self.stats["tasks_completed"]
            
            # Update shared resources if needed
            if task.specialization == AgentSpecialization.NAVIGATION and task.page_url:
                agent_info.page_context = {"url": task.page_url}
                self.resource_manager.update_page_context(
                    task.page_url, 
                    {"last_visited": asyncio.get_event_loop().time()},
                    agent_id
                )
            
            logger.info(f"Task {task.id} completed by agent {agent_id} in {execution_time:.2f}s")
            
        except Exception as e:
            # Handle task failure
            task.error = e
            task.status = "failed"
            task.completed_at = asyncio.get_event_loop().time()
            
            # Update agent performance
            self.workload_balancer.update_agent_score(agent_id, -0.2)  # Negative score for failure
            agent_info.tasks_failed += 1
            
            # Store failed task
            self.failed_tasks[task.id] = task
            
            # Update statistics
            self.stats["tasks_failed"] += 1
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = "pending"
                task.assigned_agent = None
                await self.task_queue.put((task.priority.value, task))
                logger.warning(f"Task {task.id} failed, retrying (attempt {task.retry_count})")
            else:
                logger.error(f"Task {task.id} failed permanently: {e}")
            
        finally:
            # Release resource locks
            for resource in task.shared_resources:
                self.resource_manager.release_lock(resource, agent_id)
            
            # Update agent status
            agent_info.status = "idle"
            agent_info.current_task = None
            agent_info.last_active = asyncio.get_event_loop().time()
            
            # Clean up task tracking
            if task.id in self.agent_tasks:
                del self.agent_tasks[task.id]
    
    async def _update_workload_balancing(self):
        """Update workload balancing based on current state."""
        agents = list(self.agent_pool.values())
        pending_tasks = []
        
        # Get pending tasks from queue (without removing them)
        # Note: This is a simplified approach; in production you'd want a more sophisticated method
        temp_queue = asyncio.PriorityQueue()
        while not self.task_queue.empty():
            try:
                priority, task = self.task_queue.get_nowait()
                pending_tasks.append(task)
                await temp_queue.put((priority, task))
            except asyncio.QueueEmpty:
                break
        
        # Restore queue
        self.task_queue = temp_queue
        
        # Update balancing
        if pending_tasks:
            self.workload_balancer.balance_load(agents, pending_tasks)
    
    async def _handle_conflicts(self):
        """Handle resource conflicts between agents."""
        conflicts = self.resource_manager.get_resource_conflicts()
        
        for resource_id, agent_ids in conflicts:
            if len(agent_ids) > 1:
                self.stats["conflicts_detected"] += 1
                logger.warning(f"Resource conflict detected on {resource_id} between agents {agent_ids}")
                
                # Simple conflict resolution: prioritize based on task priority
                # In a real implementation, you might have more sophisticated resolution
                for agent_id in agent_ids:
                    agent_info = self.agent_pool.get(agent_id)
                    if agent_info and agent_info.current_task:
                        # Get task from completed or failed tasks (simplified)
                        # In reality, you'd track active tasks differently
                        pass
                
                self.stats["conflicts_resolved"] += 1
    
    async def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific agent."""
        agent_info = self.agent_pool.get(agent_id)
        if not agent_info:
            return None
        
        return {
            "id": agent_info.id,
            "specialization": agent_info.specialization.value,
            "status": agent_info.status,
            "current_task": agent_info.current_task,
            "tasks_completed": agent_info.tasks_completed,
            "tasks_failed": agent_info.tasks_failed,
            "performance_score": self.workload_balancer.agent_scores.get(agent_id, 1.0),
            "last_active": agent_info.last_active
        }
    
    async def get_all_agent_statuses(self) -> List[Dict[str, Any]]:
        """Get status of all agents."""
        return [await self.get_agent_status(agent_id) for agent_id in self.agent_pool.keys()]
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        # Check completed tasks
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return {
                "id": task.id,
                "status": task.status,
                "description": task.description,
                "specialization": task.specialization.value,
                "assigned_agent": task.assigned_agent,
                "result": task.result,
                "execution_time": task.completed_at - task.started_at if task.completed_at and task.started_at else None
            }
        
        # Check failed tasks
        if task_id in self.failed_tasks:
            task = self.failed_tasks[task_id]
            return {
                "id": task.id,
                "status": task.status,
                "description": task.description,
                "specialization": task.specialization.value,
                "assigned_agent": task.assigned_agent,
                "error": str(task.error) if task.error else None,
                "retry_count": task.retry_count
            }
        
        return None
    
    async def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            **self.stats,
            "active_agents": len([a for a in self.agent_pool.values() if a.status == "busy"]),
            "total_agents": len(self.agent_pool),
            "pending_tasks": self.task_queue.qsize(),
            "resource_conflicts": len(self.resource_manager.get_resource_conflicts())
        }
    
    async def broadcast_message(self, message: Dict[str, Any], target_agents: Optional[List[str]] = None):
        """Broadcast a message to agents."""
        if target_agents is None:
            target_agents = list(self.agent_pool.keys())
        
        for agent_id in target_agents:
            if agent_id in self.message_queues:
                await self.message_queues[agent_id].put(message)
    
    async def send_message_to_agent(self, agent_id: str, message: Dict[str, Any]):
        """Send a message to a specific agent."""
        if agent_id in self.message_queues:
            await self.message_queues[agent_id].put(message)
    
    async def get_agent_messages(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get messages for a specific agent."""
        messages = []
        if agent_id in self.message_queues:
            while not self.message_queues[agent_id].empty():
                try:
                    message = self.message_queues[agent_id].get_nowait()
                    messages.append(message)
                except asyncio.QueueEmpty:
                    break
        return messages


# Convenience functions for easy integration
async def create_orchestrator(max_agents: int = 5) -> AgentOrchestrator:
    """Create and start an orchestrator."""
    orchestrator = AgentOrchestrator(max_agents=max_agents)
    await orchestrator.start()
    return orchestrator


async def submit_form_filling_task(
    orchestrator: AgentOrchestrator,
    form_url: str,
    form_data: Dict[str, Any],
    priority: TaskPriority = TaskPriority.MEDIUM
) -> str:
    """Submit a form filling task to the orchestrator."""
    task = Task(
        description=f"Fill form at {form_url}",
        specialization=AgentSpecialization.FORM_FILLING,
        priority=priority,
        parameters={
            "type": "fill_form",
            "form_url": form_url,
            "fields": form_data
        },
        page_url=form_url,
        requires_login=True
    )
    return await orchestrator.submit_task(task)


async def submit_data_extraction_task(
    orchestrator: AgentOrchestrator,
    urls: List[str],
    selectors: Dict[str, str],
    priority: TaskPriority = TaskPriority.MEDIUM
) -> str:
    """Submit a data extraction task to the orchestrator."""
    task = Task(
        description=f"Extract data from {len(urls)} URLs",
        specialization=AgentSpecialization.DATA_EXTRACTION,
        priority=priority,
        parameters={
            "type": "extract_data",
            "urls": urls,
            "selectors": selectors
        }
    )
    return await orchestrator.submit_task(task)


async def submit_navigation_task(
    orchestrator: AgentOrchestrator,
    url: str,
    wait_for: Optional[str] = None,
    priority: TaskPriority = TaskPriority.HIGH
) -> str:
    """Submit a navigation task to the orchestrator."""
    task = Task(
        description=f"Navigate to {url}",
        specialization=AgentSpecialization.NAVIGATION,
        priority=priority,
        parameters={
            "url": url,
            "wait_for": wait_for
        },
        page_url=url
    )
    return await orchestrator.submit_task(task)