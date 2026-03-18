"""Element class for element operations with intelligent recovery system and predictive prefetching."""

import asyncio
import hashlib
import json
import time
from typing import TYPE_CHECKING, Literal, Union, Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

from cdp_use.client import logger
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from cdp_use.cdp.dom.commands import (
        DescribeNodeParameters,
        FocusParameters,
        GetAttributesParameters,
        GetBoxModelParameters,
        PushNodesByBackendIdsToFrontendParameters,
        RequestChildNodesParameters,
        ResolveNodeParameters,
    )
    from cdp_use.cdp.input.commands import (
        DispatchMouseEventParameters,
    )
    from cdp_use.cdp.input.types import MouseButton
    from cdp_use.cdp.page.commands import CaptureScreenshotParameters
    from cdp_use.cdp.page.types import Viewport
    from cdp_use.cdp.runtime.commands import CallFunctionOnParameters

    from browser_use_ultra.browser.session import BrowserSession

# Type definitions for element operations
ModifierType = Literal['Alt', 'Control', 'Meta', 'Shift']


class Position(TypedDict):
    """2D position coordinates."""

    x: float
    y: float


class BoundingBox(TypedDict):
    """Element bounding box with position and dimensions."""

    x: float
    y: float
    width: float
    height: float


class ElementInfo(TypedDict):
    """Basic information about a DOM element."""

    backendNodeId: int
    nodeId: int | None
    nodeName: str
    nodeType: int
    nodeValue: str | None
    attributes: dict[str, str]
    boundingBox: BoundingBox | None
    error: str | None


class RecoveryStrategy(Enum):
    """Recovery strategy types for element recovery."""
    ORIGINAL_SELECTOR = "original_selector"
    VISUAL_SIMILARITY = "visual_similarity"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    LLM_IDENTIFICATION = "llm_identification"


class PrefetchStatus(Enum):
    """Status of prefetched elements."""
    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"
    STALE = "stale"


@dataclass
class RecoveryContext:
    """Context information for element recovery."""
    original_selector: Optional[str] = None
    selector_type: Optional[str] = None  # 'css', 'xpath', 'text', etc.
    visual_signature: Optional[Dict[str, Any]] = None
    semantic_context: Optional[Dict[str, Any]] = None
    iframe_path: List[str] = field(default_factory=list)
    shadow_dom_path: List[str] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class RecoveryPattern:
    """Cached recovery pattern for future use."""
    pattern_id: str
    original_selector: str
    recovery_selector: str
    strategy: RecoveryStrategy
    success_count: int = 0
    last_used: float = 0.0
    context_hash: str = ""


@dataclass
class PrefetchEntry:
    """Entry for prefetched element data."""
    selector: str
    selector_type: str
    element_info: Optional[ElementInfo] = None
    status: PrefetchStatus = PrefetchStatus.PENDING
    timestamp: float = 0.0
    context_hash: str = ""
    js_context_warmed: bool = False
    predicted_actions: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ActionTransition:
    """Transition in Markov model for action sequences."""
    from_action: str
    to_action: str
    count: int = 0
    probability: float = 0.0
    avg_latency: float = 0.0
    last_updated: float = 0.0


class MarkovActionModel:
    """Markov model for predicting action sequences."""
    
    def __init__(self, window_size: int = 100):
        self.transitions: Dict[Tuple[str, str], ActionTransition] = {}
        self.action_counts: Dict[str, int] = defaultdict(int)
        self.sequence_window: deque = deque(maxlen=window_size)
        self.total_sequences: int = 0
        
    def record_transition(self, from_action: str, to_action: str, latency: float = 0.0):
        """Record an action transition."""
        key = (from_action, to_action)
        
        if key not in self.transitions:
            self.transitions[key] = ActionTransition(
                from_action=from_action,
                to_action=to_action
            )
        
        transition = self.transitions[key]
        transition.count += 1
        transition.last_updated = time.time()
        
        # Update running average latency
        if transition.count == 1:
            transition.avg_latency = latency
        else:
            transition.avg_latency = (
                (transition.avg_latency * (transition.count - 1) + latency) / 
                transition.count
            )
        
        self.action_counts[from_action] += 1
        self.total_sequences += 1
        
        # Update probabilities
        self._update_probabilities(from_action)
    
    def _update_probabilities(self, from_action: str):
        """Update transition probabilities for a given action."""
        total_from = self.action_counts[from_action]
        
        for key, transition in self.transitions.items():
            if transition.from_action == from_action:
                transition.probability = transition.count / total_from
    
    def predict_next_actions(self, current_action: str, top_n: int = 3) -> List[Tuple[str, float]]:
        """Predict most likely next actions given current action."""
        predictions = []
        
        for key, transition in self.transitions.items():
            if transition.from_action == current_action:
                predictions.append((transition.to_action, transition.probability))
        
        # Sort by probability and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_n]
    
    def record_sequence(self, actions: List[str]):
        """Record a complete sequence of actions."""
        for i in range(len(actions) - 1):
            self.record_transition(actions[i], actions[i + 1])
        
        # Store sequence in window for pattern analysis
        self.sequence_window.append(tuple(actions))
    
    def get_common_patterns(self, min_support: float = 0.1) -> List[List[str]]:
        """Extract common action patterns from the sequence window."""
        pattern_counts = defaultdict(int)
        
        # Extract all subsequences of length 2-5
        for sequence in self.sequence_window:
            for length in range(2, min(6, len(sequence) + 1)):
                for i in range(len(sequence) - length + 1):
                    pattern = sequence[i:i + length]
                    pattern_counts[pattern] += 1
        
        # Filter by minimum support
        total_sequences = len(self.sequence_window)
        common_patterns = []
        
        for pattern, count in pattern_counts.items():
            support = count / total_sequences if total_sequences > 0 else 0
            if support >= min_support:
                common_patterns.append(list(pattern))
        
        return common_patterns


class PredictivePrefetcher:
    """Predictive prefetching system for anticipating agent needs."""
    
    def __init__(self, browser_session: 'BrowserSession'):
        self._browser_session = browser_session
        self._client = browser_session.cdp_client
        self._markov_model = MarkovActionModel()
        self._prefetch_cache: Dict[str, PrefetchEntry] = {}
        self._selector_validity_cache: Dict[str, bool] = {}
        self._js_context_cache: Dict[str, bool] = {}
        self._current_action: Optional[str] = None
        self._last_prefetch_time: float = 0.0
        self._prefetch_interval: float = 0.1  # Minimum time between prefetches
        
        # Warm up WebSocket connection
        self._warm_websocket_connection()
    
    async def _warm_websocket_connection(self):
        """Warm up WebSocket connection to CDP."""
        try:
            # Send a lightweight command to keep connection warm
            await self._client.send.Runtime.evaluate({
                'expression': '1 + 1',
                'returnByValue': True
            })
            logger.debug("WebSocket connection warmed up")
        except Exception as e:
            logger.debug(f"Failed to warm WebSocket connection: {e}")
    
    def _generate_cache_key(self, selector: str, selector_type: str = "css") -> str:
        """Generate cache key for selector."""
        key_data = f"{selector_type}:{selector}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def record_action(self, action: str, selector: str, latency: float = 0.0):
        """Record an action performed by the agent."""
        if self._current_action:
            # Record transition from previous action to current action
            self._markov_model.record_transition(
                self._current_action, 
                action, 
                latency
            )
        
        self._current_action = action
        
        # Trigger prefetching for likely next actions
        await self._prefetch_likely_next_elements(action, selector)
    
    async def _prefetch_likely_next_elements(self, current_action: str, current_selector: str):
        """Prefetch elements for likely next actions."""
        current_time = time.time()
        
        # Respect prefetch interval to avoid overwhelming the browser
        if current_time - self._last_prefetch_time < self._prefetch_interval:
            return
        
        self._last_prefetch_time = current_time
        
        # Get predicted next actions
        predictions = self._markov_model.predict_next_actions(current_action, top_n=3)
        
        if not predictions:
            return
        
        logger.debug(f"Prefetching for predictions: {predictions}")
        
        # Prefetch elements for each predicted action
        prefetch_tasks = []
        for action, probability in predictions:
            if probability > 0.1:  # Only prefetch if probability > 10%
                task = self._prefetch_for_action(action, current_selector, probability)
                prefetch_tasks.append(task)
        
        # Execute prefetch tasks concurrently
        if prefetch_tasks:
            await asyncio.gather(*prefetch_tasks, return_exceptions=True)
    
    async def _prefetch_for_action(self, action: str, context_selector: str, confidence: float):
        """Prefetch elements for a specific action."""
        # Generate likely selectors based on action type and context
        likely_selectors = self._generate_likely_selectors(action, context_selector)
        
        for selector_info in likely_selectors:
            selector = selector_info['selector']
            selector_type = selector_info['type']
            
            cache_key = self._generate_cache_key(selector, selector_type)
            
            # Skip if already prefetched recently
            if cache_key in self._prefetch_cache:
                entry = self._prefetch_cache[cache_key]
                if time.time() - entry.timestamp < 5.0:  # 5 second cache
                    continue
            
            # Create prefetch entry
            entry = PrefetchEntry(
                selector=selector,
                selector_type=selector_type,
                timestamp=time.time(),
                predicted_actions=[action],
                confidence=confidence
            )
            
            self._prefetch_cache[cache_key] = entry
            
            # Prefetch element info
            await self._prefetch_element_info(entry)
            
            # Precompute selector validity
            await self._precompute_selector_validity(entry)
            
            # Warm up JavaScript context
            await self._warm_js_context(entry)
    
    def _generate_likely_selectors(self, action: str, context_selector: str) -> List[Dict[str, str]]:
        """Generate likely selectors based on action type and context."""
        selectors = []
        
        # Extract context from current selector
        context_parts = context_selector.split(' ') if context_selector else []
        
        if action == "click":
            # For click actions, look for buttons, links, and clickable elements
            selectors.extend([
                {'selector': f"{context_selector} button", 'type': 'css'},
                {'selector': f"{context_selector} a", 'type': 'css'},
                {'selector': f"{context_selector} [role='button']", 'type': 'css'},
                {'selector': f"{context_selector} [onclick]", 'type': 'css'},
            ])
        
        elif action == "type":
            # For type actions, look for input fields
            selectors.extend([
                {'selector': f"{context_selector} input", 'type': 'css'},
                {'selector': f"{context_selector} textarea", 'type': 'css'},
                {'selector': f"{context_selector} [contenteditable='true']", 'type': 'css'},
            ])
        
        elif action == "select":
            # For select actions, look for dropdowns
            selectors.extend([
                {'selector': f"{context_selector} select", 'type': 'css'},
                {'selector': f"{context_selector} [role='listbox']", 'type': 'css'},
            ])
        
        # Add common navigation patterns
        selectors.extend([
            {'selector': f"{context_selector} [aria-label]", 'type': 'css'},
            {'selector': f"{context_selector} [data-testid]", 'type': 'css'},
        ])
        
        return selectors
    
    async def _prefetch_element_info(self, entry: PrefetchEntry):
        """Prefetch element information."""
        try:
            if entry.selector_type == "css":
                # Use DOM.querySelector to check if element exists
                result = await self._client.send.DOM.querySelector({
                    'nodeId': 1,  # Document node
                    'selector': entry.selector
                })
                
                if result.get('nodeId', 0) != 0:
                    # Element exists, get more info
                    node_info = await self._client.send.DOM.describeNode({
                        'nodeId': result['nodeId']
                    })
                    
                    # Get bounding box if possible
                    try:
                        box_model = await self._client.send.DOM.getBoxModel({
                            'nodeId': result['nodeId']
                        })
                        bounding_box = {
                            'x': box_model['model']['content'][0],
                            'y': box_model['model']['content'][1],
                            'width': box_model['model']['width'],
                            'height': box_model['model']['height']
                        }
                    except Exception:
                        bounding_box = None
                    
                    entry.element_info = {
                        'backendNodeId': node_info.get('backendNodeId', 0),
                        'nodeId': result['nodeId'],
                        'nodeName': node_info.get('nodeName', ''),
                        'nodeType': node_info.get('nodeType', 0),
                        'nodeValue': node_info.get('nodeValue'),
                        'attributes': dict(zip(
                            node_info.get('attributes', [])[::2],
                            node_info.get('attributes', [])[1::2]
                        )) if node_info.get('attributes') else {},
                        'boundingBox': bounding_box,
                        'error': None
                    }
                    entry.status = PrefetchStatus.VALID
                else:
                    entry.status = PrefetchStatus.INVALID
            
        except Exception as e:
            logger.debug(f"Failed to prefetch element info for {entry.selector}: {e}")
            entry.status = PrefetchStatus.INVALID
    
    async def _precompute_selector_validity(self, entry: PrefetchEntry):
        """Precompute selector validity."""
        cache_key = self._generate_cache_key(entry.selector, entry.selector_type)
        
        try:
            if entry.selector_type == "css":
                # Quick validity check
                result = await self._client.send.Runtime.evaluate({
                    'expression': f"document.querySelector('{entry.selector}') !== null",
                    'returnByValue': True
                })
                
                self._selector_validity_cache[cache_key] = result.get('result', {}).get('value', False)
        
        except Exception as e:
            logger.debug(f"Failed to precompute selector validity for {entry.selector}: {e}")
            self._selector_validity_cache[cache_key] = False
    
    async def _warm_js_context(self, entry: PrefetchEntry):
        """Warm up JavaScript context for element interaction."""
        if entry.status != PrefetchStatus.VALID or not entry.element_info:
            return
        
        cache_key = self._generate_cache_key(entry.selector, entry.selector_type)
        
        try:
            # Warm up by executing common operations that might be needed
            js_expressions = [
                # Check if element is visible
                f"(() => {{ const el = document.querySelector('{entry.selector}'); return el ? (el.offsetWidth > 0 && el.offsetHeight > 0) : false; }})()",
                # Check if element is enabled
                f"(() => {{ const el = document.querySelector('{entry.selector}'); return el ? !el.disabled : false; }})()",
                # Get element text content
                f"(() => {{ const el = document.querySelector('{entry.selector}'); return el ? el.textContent || '' : ''; }})()",
            ]
            
            for expr in js_expressions:
                await self._client.send.Runtime.evaluate({
                    'expression': expr,
                    'returnByValue': True
                })
            
            self._js_context_cache[cache_key] = True
            entry.js_context_warmed = True
            
        except Exception as e:
            logger.debug(f"Failed to warm JS context for {entry.selector}: {e}")
            self._js_context_cache[cache_key] = False
    
    async def get_prefetched_element(self, selector: str, selector_type: str = "css") -> Optional[PrefetchEntry]:
        """Get prefetched element data if available."""
        cache_key = self._generate_cache_key(selector, selector_type)
        
        if cache_key in self._prefetch_cache:
            entry = self._prefetch_cache[cache_key]
            
            # Check if entry is still fresh (less than 10 seconds old)
            if time.time() - entry.timestamp < 10.0:
                return entry
            else:
                entry.status = PrefetchStatus.STALE
        
        return None
    
    def is_selector_valid(self, selector: str, selector_type: str = "css") -> Optional[bool]:
        """Check if selector validity is precomputed."""
        cache_key = self._generate_cache_key(selector, selector_type)
        return self._selector_validity_cache.get(cache_key)
    
    def is_js_context_warmed(self, selector: str, selector_type: str = "css") -> bool:
        """Check if JavaScript context is warmed for selector."""
        cache_key = self._generate_cache_key(selector, selector_type)
        return self._js_context_cache.get(cache_key, False)
    
    def record_sequence(self, actions: List[str]):
        """Record a complete sequence of actions for model training."""
        self._markov_model.record_sequence(actions)
    
    def get_prediction_confidence(self, from_action: str, to_action: str) -> float:
        """Get confidence score for action transition."""
        key = (from_action, to_action)
        if key in self._markov_model.transitions:
            return self._markov_model.transitions[key].probability
        return 0.0
    
    def clear_stale_entries(self, max_age: float = 30.0):
        """Clear stale prefetch entries."""
        current_time = time.time()
        keys_to_remove = []
        
        for key, entry in self._prefetch_cache.items():
            if current_time - entry.timestamp > max_age:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._prefetch_cache[key]
    
    async def speculative_execute(self, action: str, selector: str) -> bool:
        """Speculatively execute an action with rollback capability."""
        # This is a placeholder for speculative execution
        # In a real implementation, this would execute the action in a sandboxed context
        # and provide rollback capabilities
        
        logger.debug(f"Speculative execution requested for {action} on {selector}")
        
        # Check if we have prefetched data
        prefetched = await self.get_prefetched_element(selector)
        
        if prefetched and prefetched.status == PrefetchStatus.VALID:
            # We have valid prefetched data, could execute speculatively
            # For now, just return success
            return True
        
        return False


class ElementRecoverySystem:
    """Intelligent element recovery system with multi-layered fallback strategies."""
    
    def __init__(self, browser_session: 'BrowserSession'):
        self._browser_session = browser_session
        self._client = browser_session.cdp_client
        self._recovery_cache: Dict[str, RecoveryPattern] = {}
        self._visual_model = None  # Placeholder for CV model
        self._llm_model = None  # Placeholder for LLM model
        self._prefetcher = PredictivePrefetcher(browser_session)
        
    def _generate_context_hash(self, context: RecoveryContext) -> str:
        """Generate hash for recovery context to use as cache key."""
        context_str = json.dumps({
            'selector': context.original_selector,
            'type': context.selector_type,
            'iframe_path': context.iframe_path,
            'shadow_dom_path': context.shadow_dom_path,
        }, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()
    
    async def _traverse_iframes(self, context: RecoveryContext) -> List[str]:
        """Traverse iframes to find element context."""
        iframe_selectors = []
        try:
            # Get all iframes in the document
            result = await self._client.send.Runtime.evaluate({
                'expression': '''
                    Array.from(document.querySelectorAll('iframe')).map(iframe => {
                        return {
                            src: iframe.src,
                            id: iframe.id,
                            name: iframe.name,
                            className: iframe.className
                        };
                    })
                ''',
                'returnByValue': True
            })
            
            if 'result' in result and 'value' in result['result']:
                iframes = result['result']['value']
                for iframe in iframes:
                    selector_parts = []
                    if iframe.get('id'):
                        selector_parts.append(f"iframe#{iframe['id']}")
                    if iframe.get('name'):
                        selector_parts.append(f"iframe[name='{iframe['name']}']")
                    if iframe.get('className'):
                        selector_parts.append(f"iframe.{iframe['className'].replace(' ', '.')}")
                    
                    if selector_parts:
                        iframe_selectors.append(selector_parts[0])
        except Exception as e:
            logger.debug(f"Failed to traverse iframes: {e}")
        
        return iframe_selectors
    
    async def _traverse_shadow_dom(self, context: RecoveryContext) -> List[str]:
        """Traverse shadow DOM to find element context."""
        shadow_selectors = []
        try:
            # Get all elements with shadow roots
            result = await self._client.send.Runtime.evaluate({
                'expression': '''
                    function findShadowHosts(root = document) {
                        const hosts = [];
                        const walker = document.createTreeWalker(
                            root,
                            NodeFilter.SHOW_ELEMENT,
                            null,
                            false
                        );
                        
                        let node;
                        while (node = walker.nextNode()) {
                            if (node.shadowRoot) {
                                hosts.push({
                                    tagName: node.tagName.toLowerCase(),
                                    id: node.id,
                                    className: node.className
                                });
                            }
                        }
                        return hosts;
                    }
                    findShadowHosts();
                ''',
                'returnByValue': True
            })
            
            if 'result' in result and 'value' in result['result']:
                shadow_hosts = result['result']['value']
                for host in shadow_hosts:
                    selector_parts = []
                    if host.get('id'):
                        selector_parts.append(f"#{host['id']}")
                    if host.get('className'):
                        selector_parts.append(f".{host['className'].replace(' ', '.')}")
                    
                    if selector_parts:
                        shadow_selectors.append(selector_parts[0])
        except Exception as e:
            logger.debug(f"Failed to traverse shadow DOM: {e}")
        
        return shadow_selectors
    
    async def _visual_similarity_match(self, context: RecoveryContext, screenshot: bytes) -> Optional[str]:
        """Find element using visual similarity matching."""
        # Placeholder for CV model integration
        # In a real implementation, this would use an embedded computer vision model
        # to find visually similar elements based on the original element's appearance
        logger.debug("Visual similarity matching not fully implemented - using fallback")
        return None
    
    async def _semantic_analysis(self, context: RecoveryContext) -> Optional[str]:
        """Analyze surrounding DOM to generate semantic selector."""
        try:
            if not context.original_selector:
                return None
            
            # Try to find element using semantic analysis
            # This is a simplified implementation
            result = await self._client.send.Runtime.evaluate({
                'expression': f'''
                    (() => {{
                        const elements = document.querySelectorAll('{context.original_selector}');
                        if (elements.length === 1) {{
                            return elements[0].outerHTML;
                        }}
                        return null;
                    }})()
                ''',
                'returnByValue': True
            })
            
            if result.get('result', {}).get('value'):
                # Element found with original selector
                return context.original_selector
            
            # Try to generate semantic selector based on element attributes
            # This would involve more sophisticated analysis in a real implementation
            return None
            
        except Exception as e:
            logger.debug(f"Semantic analysis failed: {e}")
            return None
    
    async def recover_element(self, context: RecoveryContext) -> Optional[str]:
        """Attempt to recover element using multiple strategies."""
        # Check cache first
        context_hash = self._generate_context_hash(context)
        
        for pattern_id, pattern in self._recovery_cache.items():
            if pattern.context_hash == context_hash:
                pattern.last_used = time.time()
                pattern.success_count += 1
                return pattern.recovery_selector
        
        # Try original selector first
        if context.original_selector:
            try:
                result = await self._client.send.DOM.querySelector({
                    'nodeId': 1,
                    'selector': context.original_selector
                })
                
                if result.get('nodeId', 0) != 0:
                    # Cache successful recovery
                    pattern_id = f"recovery_{context_hash[:8]}"
                    self._recovery_cache[pattern_id] = RecoveryPattern(
                        pattern_id=pattern_id,
                        original_selector=context.original_selector or "",
                        recovery_selector=context.original_selector,
                        strategy=RecoveryStrategy.ORIGINAL_SELECTOR,
                        context_hash=context_hash,
                        last_used=time.time()
                    )
                    return context.original_selector
            except Exception:
                pass
        
        # Try other recovery strategies
        recovery_strategies = [
            (RecoveryStrategy.SEMANTIC_ANALYSIS, self._semantic_analysis),
            (RecoveryStrategy.VISUAL_SIMILARITY, lambda ctx: self._visual_similarity_match(ctx, b"")),
        ]
        
        for strategy, recovery_func in recovery_strategies:
            try:
                recovered_selector = await recovery_func(context)
                if recovered_selector:
                    # Cache successful recovery
                    pattern_id = f"recovery_{context_hash[:8]}"
                    self._recovery_cache[pattern_id] = RecoveryPattern(
                        pattern_id=pattern_id,
                        original_selector=context.original_selector or "",
                        recovery_selector=recovered_selector,
                        strategy=strategy,
                        context_hash=context_hash,
                        last_used=time.time()
                    )
                    return recovered_selector
            except Exception as e:
                logger.debug(f"Recovery strategy {strategy} failed: {e}")
                continue
        
        return None
    
    def get_prefetcher(self) -> PredictivePrefetcher:
        """Get the predictive prefetcher instance."""
        return self._prefetcher
    
    async def record_action(self, action: str, selector: str, latency: float = 0.0):
        """Record an action for predictive prefetching."""
        await self._prefetcher.record_action(action, selector, latency)
    
    def record_sequence(self, actions: List[str]):
        """Record a sequence of actions for model training."""
        self._prefetcher.record_sequence(actions)