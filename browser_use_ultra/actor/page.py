"""Page class for page-level operations."""

from typing import TYPE_CHECKING, TypeVar, Dict, List, Optional, Set
from collections import defaultdict
import asyncio
import time
import json
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel

from browser_use_ultra import logger
from browser_use_ultra.actor.utils import get_key_info
from browser_use_ultra.dom.serializer.serializer import DOMTreeSerializer
from browser_use_ultra.dom.service import DomService
from browser_use_ultra.llm.messages import SystemMessage, UserMessage

T = TypeVar('T', bound=BaseModel)

if TYPE_CHECKING:
	from cdp_use.cdp.dom.commands import (
		DescribeNodeParameters,
		QuerySelectorAllParameters,
	)
	from cdp_use.cdp.emulation.commands import SetDeviceMetricsOverrideParameters
	from cdp_use.cdp.input.commands import (
		DispatchKeyEventParameters,
	)
	from cdp_use.cdp.page.commands import CaptureScreenshotParameters, NavigateParameters, NavigateToHistoryEntryParameters
	from cdp_use.cdp.runtime.commands import EvaluateParameters
	from cdp_use.cdp.target.commands import (
		AttachToTargetParameters,
		GetTargetInfoParameters,
	)
	from cdp_use.cdp.target.types import TargetInfo

	from browser_use_ultra.browser.session import BrowserSession
	from browser_use_ultra.llm.base import BaseChatModel

	from .element import Element
	from .mouse import Mouse


class ActionType(Enum):
	"""Types of actions for Markov modeling."""
	CLICK = "click"
	TYPE = "type"
	SCROLL = "scroll"
	NAVIGATE = "navigate"
	HOVER = "hover"
	SELECT = "select"
	WAIT = "wait"


@dataclass
class ActionTransition:
	"""Represents a transition between actions in Markov model."""
	from_action: ActionType
	to_action: ActionType
	probability: float = 0.0
	count: int = 0


@dataclass
class PrefetchCandidate:
	"""Candidate for prefetching."""
	selector: str
	action_type: ActionType
	confidence: float
	backend_node_id: Optional[int] = None
	element_hash: Optional[str] = None
	timestamp: float = field(default_factory=time.time)


class MarkovActionModel:
	"""Markov model for predicting next actions."""
	
	def __init__(self, window_size: int = 100):
		self.transitions: Dict[ActionType, Dict[ActionType, int]] = defaultdict(lambda: defaultdict(int))
		self.action_counts: Dict[ActionType, int] = defaultdict(int)
		self.window_size = window_size
		self.recent_actions: List[ActionType] = []
		
	def record_transition(self, from_action: ActionType, to_action: ActionType):
		"""Record an action transition."""
		self.transitions[from_action][to_action] += 1
		self.action_counts[from_action] += 1
		
		# Maintain sliding window
		self.recent_actions.append(to_action)
		if len(self.recent_actions) > self.window_size:
			old_action = self.recent_actions.pop(0)
			# Decrement count for removed action
			if old_action in self.action_counts:
				self.action_counts[old_action] = max(0, self.action_counts[old_action] - 1)
	
	def predict_next_actions(self, current_action: ActionType, top_n: int = 3) -> List[ActionTransition]:
		"""Predict most likely next actions."""
		if current_action not in self.transitions:
			return []
		
		total = sum(self.transitions[current_action].values())
		if total == 0:
			return []
		
		transitions = []
		for to_action, count in self.transitions[current_action].items():
			prob = count / total
			transitions.append(ActionTransition(
				from_action=current_action,
				to_action=to_action,
				probability=prob,
				count=count
			))
		
		# Sort by probability descending
		transitions.sort(key=lambda x: x.probability, reverse=True)
		return transitions[:top_n]


class PrefetchCache:
	"""Cache for prefetched elements and contexts."""
	
	def __init__(self, max_size: int = 50, ttl: float = 30.0):
		self.elements: Dict[str, 'Element'] = {}
		self.selector_validity: Dict[str, bool] = {}
		self.js_contexts: Dict[str, any] = {}
		self.prefetch_candidates: List[PrefetchCandidate] = []
		self.max_size = max_size
		self.ttl = ttl
		self.hits = 0
		self.misses = 0
	
	def _cleanup_expired(self):
		"""Remove expired entries."""
		current_time = time.time()
		expired_keys = []
		
		for key, candidate in enumerate(self.prefetch_candidates):
			if current_time - candidate.timestamp > self.ttl:
				expired_keys.append(key)
		
		# Remove in reverse order to maintain indices
		for key in reversed(expired_keys):
			self.prefetch_candidates.pop(key)
	
	def add_candidate(self, candidate: PrefetchCandidate):
		"""Add a prefetch candidate."""
		self._cleanup_expired()
		
		# Check if similar candidate already exists
		for existing in self.prefetch_candidates:
			if (existing.selector == candidate.selector and 
				existing.action_type == candidate.action_type):
				# Update existing candidate
				existing.confidence = max(existing.confidence, candidate.confidence)
				existing.timestamp = time.time()
				return
		
		# Add new candidate
		self.prefetch_candidates.append(candidate)
		
		# Maintain size limit
		if len(self.prefetch_candidates) > self.max_size:
			# Remove lowest confidence candidate
			self.prefetch_candidates.sort(key=lambda x: x.confidence)
			self.prefetch_candidates.pop(0)
	
	def get_element(self, selector: str) -> Optional['Element']:
		"""Get cached element by selector."""
		if selector in self.elements:
			self.hits += 1
			return self.elements[selector]
		self.misses += 1
		return None
	
	def cache_element(self, selector: str, element: 'Element'):
		"""Cache an element."""
		self.elements[selector] = element
		
		# Maintain size limit
		if len(self.elements) > self.max_size:
			# Remove oldest element (simple FIFO)
			oldest_key = next(iter(self.elements))
			del self.elements[oldest_key]
	
	def is_selector_valid(self, selector: str) -> Optional[bool]:
		"""Check if selector validity is cached."""
		return self.selector_validity.get(selector)
	
	def cache_selector_validity(self, selector: str, is_valid: bool):
		"""Cache selector validity."""
		self.selector_validity[selector] = is_valid


class Page:
	"""Page operations (tab or iframe)."""

	def __init__(
		self, browser_session: 'BrowserSession', target_id: str, session_id: str | None = None, llm: 'BaseChatModel | None' = None
	):
		self._browser_session = browser_session
		self._client = browser_session.cdp_client
		self._target_id = target_id
		self._session_id: str | None = session_id
		self._mouse: 'Mouse | None' = None

		self._llm = llm
		
		# Predictive prefetching components
		self._markov_model = MarkovActionModel()
		self._prefetch_cache = PrefetchCache()
		self._current_action: Optional[ActionType] = None
		self._prefetch_tasks: Set[asyncio.Task] = set()
		self._speculative_elements: Dict[str, 'Element'] = {}
		self._warm_js_contexts: Set[str] = set()
		
		# Track recent selectors for pattern analysis
		self._recent_selectors: List[str] = []
		self._selector_patterns: Dict[str, List[str]] = defaultdict(list)
		
		# Initialize prefetching
		self._init_prefetching()

	async def _ensure_session(self) -> str:
		"""Ensure we have a session ID for this target."""
		if not self._session_id:
			params: 'AttachToTargetParameters' = {'targetId': self._target_id, 'flatten': True}
			result = await self._client.send.Target.attachToTarget(params)
			self._session_id = result['sessionId']

			# Enable necessary domains
			import asyncio

			await asyncio.gather(
				self._client.send.Page.enable(session_id=self._session_id),
				self._client.send.DOM.enable(session_id=self._session_id),
				self._client.send.Runtime.enable(session_id=self._session_id),
				self._client.send.Network.enable(session_id=self._session_id),
			)
			
			# Warm up JavaScript contexts after session is established
			await self._warm_up_initial_contexts()

		return self._session_id

	def _init_prefetching(self):
		"""Initialize prefetching system."""
		# Common action patterns to learn
		self._common_patterns = [
			[ActionType.CLICK, ActionType.TYPE],
			[ActionType.CLICK, ActionType.CLICK],
			[ActionType.TYPE, ActionType.CLICK],
			[ActionType.SCROLL, ActionType.CLICK],
			[ActionType.HOVER, ActionType.CLICK],
		]
		
		# Pre-train with common patterns
		for pattern in self._common_patterns:
			for i in range(len(pattern) - 1):
				self._markov_model.record_transition(pattern[i], pattern[i + 1])
	
	async def _warm_up_initial_contexts(self):
		"""Warm up common JavaScript contexts."""
		common_scripts = [
			"() => document.readyState",
			"() => document.title",
			"() => window.location.href",
			"() => document.querySelectorAll('*').length",
		]
		
		for script in common_scripts:
			try:
				await self._evaluate_warmup(script)
				self._warm_js_contexts.add(script)
			except Exception as e:
				logger.debug(f"Failed to warm up context: {e}")
	
	async def _evaluate_warmup(self, script: str) -> None:
		"""Evaluate a warmup script without returning result."""
		session_id = await self._ensure_session()
		expression = f'({script})()'
		
		params: 'EvaluateParameters' = {
			'expression': expression,
			'awaitPromise': False,
			'returnByValue': False
		}
		
		try:
			await self._client.send.Runtime.evaluate(params, session_id=session_id)
		except Exception:
			pass  # Ignore warmup failures
	
	def _record_action(self, action_type: ActionType, selector: Optional[str] = None):
		"""Record an action for Markov model training."""
		if self._current_action:
			self._markov_model.record_transition(self._current_action, action_type)
		
		self._current_action = action_type
		
		# Track selector patterns
		if selector:
			self._recent_selectors.append(selector)
			if len(self._recent_selectors) > 10:
				self._recent_selectors.pop(0)
			
			# Update selector patterns
			for i in range(len(self._recent_selectors) - 1):
				current = self._recent_selectors[i]
				next_selector = self._recent_selectors[i + 1]
				if next_selector not in self._selector_patterns[current]:
					self._selector_patterns[current].append(next_selector)
	
	async def _prefetch_likely_elements(self):
		"""Prefetch elements likely to be needed next."""
		if not self._current_action:
			return
		
		# Get predictions from Markov model
		predictions = self._markov_model.predict_next_actions(self._current_action, top_n=2)
		
		for prediction in predictions:
			if prediction.probability < 0.2:  # Only prefetch if probability > 20%
				continue
			
			# Based on action type, prefetch appropriate elements
			if prediction.to_action == ActionType.CLICK:
				await self._prefetch_clickable_elements()
			elif prediction.to_action == ActionType.TYPE:
				await self._prefetch_input_elements()
			elif prediction.to_action == ActionType.SCROLL:
				await self._prefetch_scroll_containers()
	
	async def _prefetch_clickable_elements(self):
		"""Prefetch clickable elements."""
		session_id = await self._ensure_session()
		
		# Common clickable selectors
		clickable_selectors = [
			'button',
			'a[href]',
			'[role="button"]',
			'input[type="submit"]',
			'input[type="button"]',
			'.btn',
			'.button',
			'[onclick]',
		]
		
		for selector in clickable_selectors:
			# Check cache first
			if self._prefetch_cache.is_selector_valid(selector) is False:
				continue
			
			# Schedule prefetch task
			task = asyncio.create_task(self._prefetch_element_by_selector(selector, ActionType.CLICK))
			self._prefetch_tasks.add(task)
			task.add_done_callback(self._prefetch_tasks.discard)
	
	async def _prefetch_input_elements(self):
		"""Prefetch input elements."""
		session_id = await self._ensure_session()
		
		input_selectors = [
			'input:not([type="hidden"])',
			'textarea',
			'[contenteditable="true"]',
			'select',
		]
		
		for selector in input_selectors:
			if self._prefetch_cache.is_selector_valid(selector) is False:
				continue
			
			task = asyncio.create_task(self._prefetch_element_by_selector(selector, ActionType.TYPE))
			self._prefetch_tasks.add(task)
			task.add_done_callback(self._prefetch_tasks.discard)
	
	async def _prefetch_scroll_containers(self):
		"""Prefetch scrollable containers."""
		session_id = await self._ensure_session()
		
		scroll_selectors = [
			'[style*="overflow"]',
			'.scroll',
			'.scrollable',
			'[data-scroll]',
		]
		
		for selector in scroll_selectors:
			if self._prefetch_cache.is_selector_valid(selector) is False:
				continue
			
			task = asyncio.create_task(self._prefetch_element_by_selector(selector, ActionType.SCROLL))
			self._prefetch_tasks.add(task)
			task.add_done_callback(self._prefetch_tasks.discard)
	
	async def _prefetch_element_by_selector(self, selector: str, action_type: ActionType):
		"""Prefetch a single element by selector."""
		try:
			session_id = await self._ensure_session()
			
			# Check selector validity first
			document_node_id = await self._get_document_node_id()
			params: 'QuerySelectorAllParameters' = {
				'nodeId': document_node_id,
				'selector': selector
			}
			
			result = await self._client.send.DOM.querySelectorAll(params, session_id=session_id)
			
			if result['nodeIds']:
				# Cache selector validity
				self._prefetch_cache.cache_selector_validity(selector, True)
				
				# Prefetch first matching element
				backend_node_id = result['nodeIds'][0]
				element = await self.get_element(backend_node_id)
				
				# Cache the element
				self._prefetch_cache.cache_element(selector, element)
				
				# Add to prefetch candidates
				candidate = PrefetchCandidate(
					selector=selector,
					action_type=action_type,
					confidence=0.7,  # Default confidence
					backend_node_id=backend_node_id
				)
				self._prefetch_cache.add_candidate(candidate)
				
				logger.debug(f"Prefetched element: {selector}")
			else:
				# Cache that selector is invalid
				self._prefetch_cache.cache_selector_validity(selector, False)
				
		except Exception as e:
			logger.debug(f"Failed to prefetch element {selector}: {e}")
	
	async def _get_document_node_id(self) -> int:
		"""Get the document node ID."""
		session_id = await self._ensure_session()
		result = await self._client.send.DOM.getDocument({}, session_id=session_id)
		return result['root']['nodeId']
	
	async def _speculative_execute(self, selector: str, action_type: ActionType) -> Optional['Element']:
		"""Speculatively execute action preparation."""
		# Check if we have a cached element
		cached_element = self._prefetch_cache.get_element(selector)
		if cached_element:
			return cached_element
		
		# Otherwise, prefetch it
		await self._prefetch_element_by_selector(selector, action_type)
		return self._prefetch_cache.get_element(selector)
	
	def _rollback_speculative(self, selector: str):
		"""Rollback speculative execution for a selector."""
		if selector in self._speculative_elements:
			del self._speculative_elements[selector]
	
	async def get_element_with_prefetch(self, selector: str, action_type: ActionType = ActionType.CLICK) -> 'Element':
		"""Get element with predictive prefetching."""
		# Record action for Markov model
		self._record_action(action_type, selector)
		
		# Check cache first
		cached_element = self._prefetch_cache.get_element(selector)
		if cached_element:
			logger.debug(f"Cache hit for selector: {selector}")
			return cached_element
		
		# Speculatively prefetch next likely elements
		asyncio.create_task(self._prefetch_likely_elements())
		
		# Get the requested element
		element = await self.get_element_by_selector(selector)
		
		# Cache it
		self._prefetch_cache.cache_element(selector, element)
		
		return element
	
	async def get_element_by_selector(self, selector: str) -> 'Element':
		"""Get element by selector (with caching)."""
		session_id = await self._ensure_session()
		
		# Check selector validity cache
		is_valid = self._prefetch_cache.is_selector_valid(selector)
		if is_valid is False:
			raise ValueError(f"Selector {selector} is invalid (cached)")
		
		# Get document node
		document_node_id = await self._get_document_node_id()
		
		# Query selector
		params: 'QuerySelectorAllParameters' = {
			'nodeId': document_node_id,
			'selector': selector
		}
		
		result = await self._client.send.DOM.querySelectorAll(params, session_id=session_id)
		
		if not result['nodeIds']:
			self._prefetch_cache.cache_selector_validity(selector, False)
			raise ValueError(f"No element found for selector: {selector}")
		
		# Cache selector validity
		self._prefetch_cache.cache_selector_validity(selector, True)
		
		# Get first matching element
		backend_node_id = result['nodeIds'][0]
		element = await self.get_element(backend_node_id)
		
		return element
	
	def get_prefetch_stats(self) -> Dict:
		"""Get prefetching statistics."""
		cache_hit_rate = 0.0
		if self._prefetch_cache.hits + self._prefetch_cache.misses > 0:
			cache_hit_rate = self._prefetch_cache.hits / (self._prefetch_cache.hits + self._prefetch_cache.misses)
		
		return {
			'cache_hits': self._prefetch_cache.hits,
			'cache_misses': self._prefetch_cache.misses,
			'cache_hit_rate': cache_hit_rate,
			'prefetch_candidates': len(self._prefetch_cache.prefetch_candidates),
			'cached_elements': len(self._prefetch_cache.elements),
			'active_prefetch_tasks': len(self._prefetch_tasks),
			'warm_js_contexts': len(self._warm_js_contexts),
		}
	
	def clear_prefetch_cache(self):
		"""Clear all prefetch caches."""
		self._prefetch_cache = PrefetchCache()
		self._speculative_elements.clear()
		self._recent_selectors.clear()
		self._selector_patterns.clear()
	
	@property
	async def session_id(self) -> str:
		"""Get the session ID for this target.

		@dev Pass this to an arbitrary CDP call
		"""
		return await self._ensure_session()

	@property
	async def mouse(self) -> 'Mouse':
		"""Get the mouse interface for this target."""
		if not self._mouse:
			session_id = await self._ensure_session()
			from .mouse import Mouse

			self._mouse = Mouse(self._browser_session, session_id, self._target_id)
		return self._mouse

	async def reload(self) -> None:
		"""Reload the target."""
		session_id = await self._ensure_session()
		await self._client.send.Page.reload(session_id=session_id)
		# Clear prefetch cache on reload
		self.clear_prefetch_cache()

	async def get_element(self, backend_node_id: int) -> 'Element':
		"""Get an element by its backend node ID."""
		session_id = await self._ensure_session()

		from .element import Element as Element_

		return Element_(self._browser_session, backend_node_id, session_id)

	async def evaluate(self, page_function: str, *args) -> str:
		"""Execute JavaScript in the target.

		Args:
			page_function: JavaScript code that MUST start with (...args) => format
			*args: Arguments to pass to the function

		Returns:
			String representation of the JavaScript execution result.
			Objects and arrays are JSON-stringified.
		"""
		session_id = await self._ensure_session()

		# Clean and fix common JavaScript string parsing issues
		page_function = self._fix_javascript_string(page_function)

		# Enforce arrow function format
		if not (page_function.startswith('(') and '=>' in page_function):
			raise ValueError(f'JavaScript code must start with (...args) => format. Got: {page_function[:50]}...')

		# Build the expression - call the arrow function with provided args
		if args:
			# Convert args to JSON representation for safe passing
			import json

			arg_strs = [json.dumps(arg) for arg in args]
			expression = f'({page_function})({", ".join(arg_strs)})'
		else:
			expression = f'({page_function})()'

		# Debug: log the actual expression being evaluated
		logger.debug(f'Evaluating JavaScript: {repr(expression)}')

		params: 'EvaluateParameters' = {'expression': expression, 'returnByValue': True, 'awaitPromise': True}
		result = await self._client.send.Runtime.evaluate(
			params,
			session_id=session_id,
		)

		if 'exceptionDetails' in result:
			raise RuntimeError(f'JavaScript evaluation failed: {result["exceptionDetails"]}')

		value = result.get('result', {}).get('value')

		# Always return string representation
		if value is None:
			return ''
		elif isinstance(value, str):
			return value
		else:
			# Convert objects, numbers, booleans to string
			import json

			try:
				return json.dumps(value) if isinstance(value, (dict, list)) else str(value)
			except (TypeError, ValueError):
				return str(value)

	def _fix_javascript_string(self, js_code: str) -> str:
		"""Fix common JavaScript string parsing issues when written as Python string."""

		# Just do minimal, safe cleaning
		js_code = js_code.strip()

		# Only fix the most common and safe issues:

		# 1. Remove obvious Python string wrapper quotes if they exist
		if (js_code.startswith('"') and js_code.endswith('"')) or (js_code.startswith("'") and js_code.endswith("'")):
			# Check if it's a wrapped string (not part of JS syntax)
			inner = js_code[1:-1]
			if inner.count('"') + inner.count("'") == 0 or '() =>' in inner:
				js_code = inner

		# 2. Only fix clearly escaped quotes that shouldn't be
		# But be very conservative - only if we're sure it's a Python string artifact
		if '\\"' in js_code and js_code.count('\\"') > js_code.count('"'):
			js_code = js_code.replace('\\"', '"')
		if "\\'" in js_code and js_code.count("\\'") > js_code.count("'"):
			js_code = js_code.replace("\\'", "'")

		# 3. Basic whitespace normalization only
		js_code = js_code.strip()

		# Final validation - ensure it's not empty
		if not js_code:
			raise ValueError('JavaScript code is empty after cleaning')

		return js_code

	async def screenshot(self, format: str = 'png', quality: int | None = None) -> str:
		"""Take a screenshot and return base64 encoded image.

		Args:
		    format: Image format ('jpeg', 'png', 'webp')
		    quality: Quality 0-100 for JPEG format

		Returns:
		    Base64-encoded image data
		"""
		session_id = await self._ensure_session()

		params: 'CaptureScreenshotParameters' = {'format': format}

		if quality is not None and format.lower() == 'jpeg':
			params['quality'] = quality

		result = await self._client.send.Page.captureScreenshot(params, session_id=session_id)

		return result['data']

	async def press(self, key: str) -> None:
		"""Press a key on the page (sends keyboard input to the focused element or page)."""
		session_id = await self._ensure_session()

		# Handle key combinations like "Control+A"
		if '+' in key:
			parts = key.split('+')
			modifiers = parts[:-1]
			main_key = parts[-1]

			# Calculate modifier bitmask
			modifier_value = 0
			modifier_map = {'Alt': 1, 'Control': 2, 'Meta': 4, 'Shift': 8}
			for mod in modifiers:
				modifier_value |= modifier_map.get(mod, 0)

			# Press modifier keys
			for mod in modifiers:
				code, vk_code = get_key_info(mod)
				params: 'DispatchKeyEventParameters' = {'type': 'keyDown', 'key': mod, 'code': code}
				if vk_code is not None:
					params['windowsVirtualKeyCode'] = vk_code
				await self._client.send.Input.dispatchKeyEvent(params, session_id=session_id)

			# Press main key with modifiers bitmask
			main_code, main_vk_code = get_key_info(main_key)
			main_down_params: 'DispatchKeyEventParameters' = {
				'type': 'keyDown',
				'key': main_key,
				'code': main_code,
				'modifiers': modifier_value,
			}
			if main_vk_code is not None:
				main_down_params['windowsVirtualKeyCode'] = main_vk_code
			await self._client.send.Input.dispatchKeyEvent(main_down_params, session_id=session_id)

			# Release main key
			main_up_params: 'DispatchKeyEventParameters' = {
				'type': 'keyUp',
				'key': main_key,
				'code': main_code,
				'modifiers': modifier_value,
			}
			if main_vk_code is not None:
				main_up_params['windowsVirtualKeyCode'] = main_vk_code
			await self._client.send.Input.dispatchKeyEvent(main_up_params, session_id=session_id)

			# Release modifier keys
			for mod in reversed(modifiers):
				code, vk_code = get_key_info(mod)
				params: 'DispatchKeyEventParameters' = {'type': 'keyUp', 'key': mod, 'code': code}
				if vk_code is not None:
					params['windowsVirtualKeyCode'] = vk_code
				await self._client.send.Input.dispatchKeyEvent(params, session_id=session_id)
		else:
			# Single key press
			code, vk_code = get_key_info(key)
			
			# Key down
			down_params: 'DispatchKeyEventParameters' = {'type': 'keyDown', 'key': key, 'code': code}
			if vk_code is not None:
				down_params['windowsVirtualKeyCode'] = vk_code
			await self._client.send.Input.dispatchKeyEvent(down_params, session_id=session_id)

			# Key up
			up_params: 'DispatchKeyEventParameters' = {'type': 'keyUp', 'key': key, 'code': code}
			if vk_code is not None:
				up_params['windowsVirtualKeyCode'] = vk_code
			await self._client.send.Input.dispatchKeyEvent(up_params, session_id=session_id)