"""
Intelligent Element Recovery System for browser-use-ultra
Multi-layered element recovery combining visual recognition, DOM analysis, and heuristic fallbacks.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path

import numpy as np
from PIL import Image

from browser_use_ultra.actor.element import Element
from browser_use_ultra.actor.page import Page
from browser_use_ultra.actor.utils import (
    calculate_similarity,
    extract_element_features,
    get_element_screenshot,
    get_page_screenshot,
    parse_selector,
)


class RecoveryStrategy(Enum):
    """Recovery strategies in order of execution."""
    ORIGINAL_SELECTOR = "original_selector"
    VISUAL_SIMILARITY = "visual_similarity"
    DOM_SEMANTICS = "dom_semantics"
    LLM_IDENTIFICATION = "llm_identification"
    HEURISTIC_FALLBACK = "heuristic_fallback"


@dataclass
class RecoveryPattern:
    """Cached recovery pattern for future use."""
    original_selector: str
    recovery_selector: str
    strategy: RecoveryStrategy
    confidence: float
    timestamp: float
    page_signature: str
    iframe_context: Optional[str] = None
    shadow_dom_path: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self, ttl: int = 3600) -> bool:
        """Check if pattern has expired (default 1 hour TTL)."""
        return time.time() - self.timestamp > ttl


@dataclass
class VisualFeature:
    """Visual feature representation for element matching."""
    screenshot: np.ndarray
    bounding_box: Tuple[int, int, int, int]
    color_histogram: np.ndarray
    text_content: Optional[str] = None
    element_type: Optional[str] = None
    attributes: Dict[str, str] = field(default_factory=dict)


class VisualMatcher:
    """
    Intelligent Element Recovery System with multi-layered recovery strategies.
    
    Pipeline:
    1. Original selector attempt
    2. Visual similarity matching using computer vision
    3. Semantic DOM analysis
    4. LLM-powered element identification
    5. Heuristic fallbacks with iframe/shadow DOM traversal
    """
    
    def __init__(
        self,
        page: Page,
        cache_dir: Optional[str] = None,
        visual_similarity_threshold: float = 0.85,
        max_iframe_depth: int = 3,
        enable_caching: bool = True
    ):
        """
        Initialize the VisualMatcher.
        
        Args:
            page: Page instance to operate on
            cache_dir: Directory for caching recovery patterns
            visual_similarity_threshold: Threshold for visual matching (0-1)
            max_iframe_depth: Maximum depth for iframe traversal
            enable_caching: Whether to enable pattern caching
        """
        self.page = page
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".browser_use_ultra" / "recovery_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.visual_similarity_threshold = visual_similarity_threshold
        self.max_iframe_depth = max_iframe_depth
        self.enable_caching = enable_caching
        
        # Cache for recovery patterns
        self.pattern_cache: Dict[str, List[RecoveryPattern]] = {}
        self._load_cache()
        
        # Visual feature cache
        self.visual_cache: Dict[str, VisualFeature] = {}
        
        # Statistics
        self.stats = {
            "total_recoveries": 0,
            "successful_recoveries": 0,
            "strategy_success": {strategy.value: 0 for strategy in RecoveryStrategy},
            "cache_hits": 0
        }
    
    def _load_cache(self) -> None:
        """Load recovery patterns from cache file."""
        cache_file = self.cache_dir / "recovery_patterns.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    for selector, patterns in data.items():
                        self.pattern_cache[selector] = [
                            RecoveryPattern(**p) for p in patterns
                            if not RecoveryPattern(**p).is_expired()
                        ]
            except (json.JSONDecodeError, KeyError):
                self.pattern_cache = {}
    
    def _save_cache(self) -> None:
        """Save recovery patterns to cache file."""
        if not self.enable_caching:
            return
            
        cache_file = self.cache_dir / "recovery_patterns.json"
        try:
            # Convert patterns to serializable format
            serializable = {}
            for selector, patterns in self.pattern_cache.items():
                serializable[selector] = [
                    {
                        "original_selector": p.original_selector,
                        "recovery_selector": p.recovery_selector,
                        "strategy": p.strategy.value,
                        "confidence": p.confidence,
                        "timestamp": p.timestamp,
                        "page_signature": p.page_signature,
                        "iframe_context": p.iframe_context,
                        "shadow_dom_path": p.shadow_dom_path,
                        "metadata": p.metadata
                    }
                    for p in patterns
                    if not p.is_expired()
                ]
            
            with open(cache_file, 'w') as f:
                json.dump(serializable, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save recovery cache: {e}")
    
    def _get_page_signature(self) -> str:
        """Generate a signature for the current page state."""
        url = self.page.url or ""
        title = self.page.title or ""
        content_hash = hashlib.md5(
            (url + title).encode()
        ).hexdigest()
        return content_hash
    
    async def _traverse_iframes(
        self,
        selector: str,
        depth: int = 0,
        parent_context: Optional[str] = None
    ) -> List[Tuple[Element, str]]:
        """
        Recursively traverse iframes to find elements.
        
        Returns:
            List of (element, iframe_context) tuples
        """
        if depth > self.max_iframe_depth:
            return []
        
        results = []
        
        # Try in current context
        try:
            element = await self.page.query_selector(selector)
            if element:
                results.append((element, parent_context or "main"))
        except Exception:
            pass
        
        # Get all iframes
        try:
            iframes = await self.page.query_selector_all("iframe, frame")
            for i, iframe in enumerate(iframes):
                try:
                    # Switch to iframe context
                    frame = await iframe.content_frame()
                    if frame:
                        frame_context = f"{parent_context or 'main'}->iframe[{i}]"
                        
                        # Create a temporary page for the frame
                        frame_page = Page(frame)
                        frame_matcher = VisualMatcher(
                            frame_page,
                            cache_dir=str(self.cache_dir),
                            visual_similarity_threshold=self.visual_similarity_threshold,
                            max_iframe_depth=self.max_iframe_depth - depth - 1,
                            enable_caching=False
                        )
                        
                        # Recursively search in iframe
                        frame_results = await frame_matcher._traverse_iframes(
                            selector,
                            depth + 1,
                            frame_context
                        )
                        results.extend(frame_results)
                except Exception:
                    continue
        except Exception:
            pass
        
        return results
    
    async def _traverse_shadow_dom(
        self,
        selector: str,
        element: Element,
        path: Optional[List[str]] = None
    ) -> List[Tuple[Element, List[str]]]:
        """
        Traverse shadow DOM to find elements.
        
        Returns:
            List of (element, shadow_dom_path) tuples
        """
        if path is None:
            path = []
        
        results = []
        
        # Check if element has shadow root
        try:
            has_shadow = await element.evaluate("el => !!el.shadowRoot")
            if has_shadow:
                shadow_root = await element.evaluate_handle("el => el.shadowRoot")
                shadow_page = Page(shadow_root)
                shadow_matcher = VisualMatcher(
                    shadow_page,
                    cache_dir=str(self.cache_dir),
                    visual_similarity_threshold=self.visual_similarity_threshold,
                    max_iframe_depth=self.max_iframe_depth,
                    enable_caching=False
                )
                
                # Try selector in shadow DOM
                shadow_element = await shadow_page.query_selector(selector)
                if shadow_element:
                    results.append((shadow_element, path + ["shadow"]))
                
                # Recursively check for nested shadow DOMs
                shadow_elements = await shadow_page.query_selector_all("*")
                for i, shadow_el in enumerate(shadow_elements):
                    nested_results = await shadow_matcher._traverse_shadow_dom(
                        selector,
                        shadow_el,
                        path + [f"shadow[{i}]"]
                    )
                    results.extend(nested_results)
        except Exception:
            pass
        
        return results
    
    async def _original_selector_recovery(self, selector: str) -> Optional[Element]:
        """Attempt recovery using the original selector with iframe/shadow DOM traversal."""
        # Try direct selector first
        try:
            element = await self.page.query_selector(selector)
            if element:
                return element
        except Exception:
            pass
        
        # Traverse iframes
        iframe_results = await self._traverse_iframes(selector)
        if iframe_results:
            return iframe_results[0][0]
        
        # Traverse shadow DOM for all elements
        try:
            all_elements = await self.page.query_selector_all("*")
            for element in all_elements:
                shadow_results = await self._traverse_shadow_dom(selector, element)
                if shadow_results:
                    return shadow_results[0][0]
        except Exception:
            pass
        
        return None
    
    async def _extract_visual_features(self, element: Element) -> VisualFeature:
        """Extract visual features from an element."""
        try:
            # Get element screenshot
            screenshot = await get_element_screenshot(element)
            
            # Get bounding box
            bounding_box = await element.bounding_box()
            if not bounding_box:
                raise ValueError("Element has no bounding box")
            
            # Calculate color histogram
            if screenshot.size > 0:
                hist_r = np.histogram(screenshot[:,:,0], bins=8, range=(0, 256))[0]
                hist_g = np.histogram(screenshot[:,:,1], bins=8, range=(0, 256))[0]
                hist_b = np.histogram(screenshot[:,:,2], bins=8, range=(0, 256))[0]
                color_histogram = np.concatenate([hist_r, hist_g, hist_b]).astype(float)
                color_histogram /= color_histogram.sum() + 1e-10
            else:
                color_histogram = np.zeros(24)
            
            # Get text content
            text_content = await element.text_content()
            
            # Get element type
            tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
            
            # Get attributes
            attributes = await element.evaluate("""el => {
                const attrs = {};
                for (const attr of el.attributes) {
                    attrs[attr.name] = attr.value;
                }
                return attrs;
            }""")
            
            return VisualFeature(
                screenshot=screenshot,
                bounding_box=(bounding_box["x"], bounding_box["y"], 
                             bounding_box["width"], bounding_box["height"]),
                color_histogram=color_histogram,
                text_content=text_content,
                element_type=tag_name,
                attributes=attributes
            )
        except Exception as e:
            # Return empty feature on error
            return VisualFeature(
                screenshot=np.array([]),
                bounding_box=(0, 0, 0, 0),
                color_histogram=np.zeros(24)
            )
    
    async def _visual_similarity_match(
        self,
        selector: str,
        reference_features: Optional[VisualFeature] = None
    ) -> Optional[Tuple[Element, float]]:
        """
        Find element using visual similarity matching.
        
        Returns:
            Tuple of (element, similarity_score) or None
        """
        if reference_features is None:
            # Try to get reference from cache
            cache_key = f"visual_{selector}"
            if cache_key in self.visual_cache:
                reference_features = self.visual_cache[cache_key]
            else:
                return None
        
        try:
            # Get all visible elements
            elements = await self.page.query_selector_all("*")
            best_match = None
            best_score = 0.0
            
            for element in elements:
                try:
                    # Skip invisible elements
                    is_visible = await element.is_visible()
                    if not is_visible:
                        continue
                    
                    # Extract features
                    element_features = await self._extract_visual_features(element)
                    
                    # Calculate similarity
                    if element_features.screenshot.size > 0 and reference_features.screenshot.size > 0:
                        # Use multiple similarity metrics
                        visual_sim = calculate_similarity(
                            reference_features.screenshot,
                            element_features.screenshot
                        )
                        
                        # Color histogram similarity
                        color_sim = 1.0 - np.linalg.norm(
                            reference_features.color_histogram - element_features.color_histogram
                        )
                        
                        # Text similarity (if available)
                        text_sim = 0.0
                        if reference_features.text_content and element_features.text_content:
                            text_sim = 1.0 if reference_features.text_content == element_features.text_content else 0.0
                        
                        # Combined score
                        score = 0.5 * visual_sim + 0.3 * color_sim + 0.2 * text_sim
                        
                        if score > best_score and score >= self.visual_similarity_threshold:
                            best_score = score
                            best_match = element
                except Exception:
                    continue
            
            if best_match:
                return (best_match, best_score)
        except Exception:
            pass
        
        return None
    
    async def _dom_semantic_analysis(self, selector: str) -> Optional[Element]:
        """
        Analyze DOM semantics to find element.
        Looks for elements with similar attributes, structure, or context.
        """
        try:
            parsed = parse_selector(selector)
            if not parsed:
                return None
            
            # Extract key attributes from selector
            tag_name = parsed.get("tag", "")
            attributes = parsed.get("attributes", {})
            classes = parsed.get("classes", [])
            id_attr = parsed.get("id", "")
            
            # Build XPath or CSS selector based on available info
            candidates = []
            
            # Try by ID first
            if id_attr:
                element = await self.page.query_selector(f"#{id_attr}")
                if element:
                    candidates.append((element, 1.0))
            
            # Try by tag and attributes
            if tag_name:
                attr_selector = tag_name
                for attr, value in attributes.items():
                    if value:
                        attr_selector += f'[{attr}="{value}"]'
                    else:
                        attr_selector += f'[{attr}]'
                
                elements = await self.page.query_selector_all(attr_selector)
                for element in elements:
                    # Calculate confidence based on matching attributes
                    confidence = 0.0
                    element_attrs = await element.evaluate("""el => {
                        const attrs = {};
                        for (const attr of el.attributes) {
                            attrs[attr.name] = attr.value;
                        }
                        return attrs;
                    }""")
                    
                    # Score based on attribute matches
                    for attr, value in attributes.items():
                        if attr in element_attrs:
                            if value and element_attrs[attr] == value:
                                confidence += 0.2
                            elif not value:
                                confidence += 0.1
                    
                    # Bonus for matching classes
                    if classes:
                        element_classes = await element.evaluate("el => el.className")
                        if element_classes:
                            class_list = element_classes.split()
                            matching_classes = set(class_list) & set(classes)
                            confidence += 0.1 * len(matching_classes) / max(len(classes), 1)
                    
                    if confidence > 0.3:
                        candidates.append((element, confidence))
            
            # Sort by confidence and return best
            candidates.sort(key=lambda x: x[1], reverse=True)
            if candidates:
                return candidates[0][0]
        except Exception:
            pass
        
        return None
    
    async def _llm_identification(
        self,
        selector: str,
        description: Optional[str] = None
    ) -> Optional[Element]:
        """
        Use LLM to identify element from screenshot context.
        This is a placeholder for integration with LLM services.
        """
        # This would integrate with the existing LLM infrastructure
        # For now, return None as this requires external LLM service
        return None
    
    async def _heuristic_fallback(self, selector: str) -> Optional[Element]:
        """
        Apply heuristic fallbacks when other methods fail.
        Uses various heuristics to find similar elements.
        """
        heuristics = [
            self._try_partial_selector,
            self._try_attribute_substring,
            self._try_text_content_match,
            self._try_position_based_match,
            self._try_role_based_match
        ]
        
        for heuristic in heuristics:
            try:
                element = await heuristic(selector)
                if element:
                    return element
            except Exception:
                continue
        
        return None
    
    async def _try_partial_selector(self, selector: str) -> Optional[Element]:
        """Try matching with partial selector."""
        try:
            # Try without pseudo-classes
            clean_selector = selector.split(":")[0]
            if clean_selector != selector:
                return await self.page.query_selector(clean_selector)
            
            # Try with wildcard for dynamic parts
            if "[" in selector and "]" in selector:
                # Replace attribute values with wildcards
                import re
                pattern = re.compile(r'\[([^=]+)="[^"]*"\]')
                wildcard_selector = pattern.sub(r'[\1]', selector)
                if wildcard_selector != selector:
                    return await self.page.query_selector(wildcard_selector)
        except Exception:
            pass
        return None
    
    async def _try_attribute_substring(self, selector: str) -> Optional[Element]:
        """Try matching attributes by substring."""
        try:
            parsed = parse_selector(selector)
            if not parsed or not parsed.get("attributes"):
                return None
            
            elements = await self.page.query_selector_all(parsed.get("tag", "*"))
            for element in elements:
                element_attrs = await element.evaluate("""el => {
                    const attrs = {};
                    for (const attr of el.attributes) {
                        attrs[attr.name] = attr.value;
                    }
                    return attrs;
                }""")
                
                # Check if any selector attribute is a substring of element attribute
                for sel_attr, sel_value in parsed["attributes"].items():
                    if sel_attr in element_attrs:
                        if sel_value and sel_value in element_attrs[sel_attr]:
                            return element
        except Exception:
            pass
        return None
    
    async def _try_text_content_match(self, selector: str) -> Optional[Element]:
        """Try matching by text content."""
        try:
            # Extract potential text from selector
            import re
            text_match = re.search(r':contains\("([^"]+)"\)', selector)
            if text_match:
                text = text_match.group(1)
                elements = await self.page.query_selector_all("*")
                for element in elements:
                    element_text = await element.text_content()
                    if element_text and text in element_text:
                        return element
        except Exception:
            pass
        return None
    
    async def _try_position_based_match(self, selector: str) -> Optional[Element]:
        """Try matching based on position relative to other elements."""
        # This would require more context about the page structure
        return None
    
    async def _try_role_based_match(self, selector: str) -> Optional[Element]:
        """Try matching by ARIA roles and accessibility attributes."""
        try:
            # Look for elements with similar roles
            elements = await self.page.query_selector_all("[role], [aria-label], [aria-labelledby]")
            for element in elements:
                role = await element.get_attribute("role")
                aria_label = await element.get_attribute("aria-label")
                
                # Simple heuristic: if selector mentions button, look for button role
                if "button" in selector.lower() and role == "button":
                    return element
                if "link" in selector.lower() and role == "link":
                    return element
                if "input" in selector.lower() and role in ["textbox", "combobox", "searchbox"]:
                    return element
        except Exception:
            pass
        return None
    
    async def recover_element(
        self,
        selector: str,
        description: Optional[str] = None,
        reference_screenshot: Optional[np.ndarray] = None
    ) -> Optional[Element]:
        """
        Main recovery pipeline for finding elements.
        
        Args:
            selector: Original CSS selector or XPath
            description: Optional text description of the element
            reference_screenshot: Optional reference screenshot for visual matching
            
        Returns:
            Recovered Element or None if not found
        """
        self.stats["total_recoveries"] += 1
        
        # Check cache first
        if self.enable_caching:
            cache_key = selector
            if cache_key in self.pattern_cache:
                page_sig = self._get_page_signature()
                for pattern in self.pattern_cache[cache_key]:
                    if pattern.page_signature == page_sig and not pattern.is_expired():
                        try:
                            element = await self.page.query_selector(pattern.recovery_selector)
                            if element:
                                self.stats["cache_hits"] += 1
                                self.stats["successful_recoveries"] += 1
                                self.stats["strategy_success"][pattern.strategy.value] += 1
                                return element
                        except Exception:
                            continue
        
        # Recovery pipeline
        recovery_strategies = [
            (RecoveryStrategy.ORIGINAL_SELECTOR, self._original_selector_recovery),
            (RecoveryStrategy.VISUAL_SIMILARITY, lambda s: self._visual_similarity_match(s)),
            (RecoveryStrategy.DOM_SEMANTICS, self._dom_semantic_analysis),
            (RecoveryStrategy.LLM_IDENTIFICATION, lambda s: self._llm_identification(s, description)),
            (RecoveryStrategy.HEURISTIC_FALLBACK, self._heuristic_fallback)
        ]
        
        # Store reference features for visual matching
        reference_features = None
        if reference_screenshot is not None:
            reference_features = VisualFeature(
                screenshot=reference_screenshot,
                bounding_box=(0, 0, 0, 0),
                color_histogram=np.zeros(24)
            )
            self.visual_cache[f"visual_{selector}"] = reference_features
        
        for strategy, recovery_func in recovery_strategies:
            try:
                result = await recovery_func(selector)
                
                if result:
                    # For visual similarity, we get a tuple (element, score)
                    if strategy == RecoveryStrategy.VISUAL_SIMILARITY and isinstance(result, tuple):
                        element, confidence = result
                    else:
                        element = result
                        confidence = 0.9  # High confidence for non-visual strategies
                    
                    # Cache successful recovery pattern
                    if self.enable_caching and element:
                        recovery_selector = await self._generate_selector(element)
                        pattern = RecoveryPattern(
                            original_selector=selector,
                            recovery_selector=recovery_selector,
                            strategy=strategy,
                            confidence=confidence,
                            timestamp=time.time(),
                            page_signature=self._get_page_signature()
                        )
                        
                        if selector not in self.pattern_cache:
                            self.pattern_cache[selector] = []
                        self.pattern_cache[selector].append(pattern)
                        
                        # Limit cache size
                        if len(self.pattern_cache[selector]) > 10:
                            self.pattern_cache[selector] = self.pattern_cache[selector][-10:]
                        
                        self._save_cache()
                    
                    self.stats["successful_recoveries"] += 1
                    self.stats["strategy_success"][strategy.value] += 1
                    return element
            except Exception as e:
                continue
        
        return None
    
    async def _generate_selector(self, element: Element) -> str:
        """Generate a unique selector for an element."""
        try:
            # Try to generate a unique CSS selector
            selector = await element.evaluate("""el => {
                function getSelector(el) {
                    if (el.id) {
                        return '#' + el.id;
                    }
                    
                    if (el === document.body) {
                        return 'body';
                    }
                    
                    let selector = el.tagName.toLowerCase();
                    
                    if (el.className && typeof el.className === 'string') {
                        selector += '.' + el.className.trim().replace(/\\s+/g, '.');
                    }
                    
                    // Add attributes for more specificity
                    const importantAttrs = ['name', 'type', 'role', 'aria-label'];
                    for (const attr of importantAttrs) {
                        if (el.hasAttribute(attr)) {
                            selector += `[${attr}="${el.getAttribute(attr)}"]`;
                        }
                    }
                    
                    // Add position among siblings
                    const parent = el.parentNode;
                    if (parent) {
                        const siblings = Array.from(parent.children).filter(
                            child => child.tagName === el.tagName
                        );
                        if (siblings.length > 1) {
                            const index = siblings.indexOf(el) + 1;
                            selector += ':nth-of-type(' + index + ')';
                        }
                    }
                    
                    return selector;
                }
                
                return getSelector(el);
            }""")
            
            return selector
        except Exception:
            # Fallback to tag name
            tag = await element.evaluate("el => el.tagName.toLowerCase()")
            return tag
    
    def get_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        return {
            **self.stats,
            "cache_size": sum(len(patterns) for patterns in self.pattern_cache.values()),
            "visual_cache_size": len(self.visual_cache)
        }
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self.pattern_cache.clear()
        self.visual_cache.clear()
        cache_file = self.cache_dir / "recovery_patterns.json"
        if cache_file.exists():
            cache_file.unlink()


# Integration with existing Page class
async def enhanced_find_element(
    page: Page,
    selector: str,
    description: Optional[str] = None,
    timeout: int = 5000
) -> Optional[Element]:
    """
    Enhanced element finding with visual recovery.
    
    This function can be used as a drop-in replacement for page.query_selector
    with intelligent recovery capabilities.
    """
    # Try direct query first
    try:
        element = await page.query_selector(selector)
        if element:
            return element
    except Exception:
        pass
    
    # If direct query fails, use visual matcher
    matcher = VisualMatcher(page)
    
    # Take screenshot for visual reference
    try:
        screenshot = await get_page_screenshot(page)
        element = await matcher.recover_element(
            selector,
            description=description,
            reference_screenshot=screenshot
        )
        return element
    except Exception:
        return None


# Export for use in other modules
__all__ = [
    "VisualMatcher",
    "RecoveryStrategy",
    "RecoveryPattern",
    "VisualFeature",
    "enhanced_find_element"
]