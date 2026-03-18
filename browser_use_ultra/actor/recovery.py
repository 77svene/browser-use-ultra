"""
Intelligent Element Recovery System for browser-use-ultra.

This module provides a multi-layered element recovery system that combines visual recognition,
DOM analysis, and heuristic fallbacks when primary selectors fail. It includes automatic iframe
traversal, shadow DOM penetration, and dynamic selector generation based on visual context.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from browser_use_ultra.actor.element import Element
from browser_use_ultra.actor.page import Page
from browser_use_ultra.actor.utils import (
    calculate_similarity_score,
    extract_element_features,
    generate_xpath,
    get_element_screenshot,
    parse_selector,
)

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Available recovery strategies in order of execution."""
    ORIGINAL_SELECTOR = "original_selector"
    VISUAL_SIMILARITY = "visual_similarity"
    DOM_SEMANTICS = "dom_semantics"
    LLM_IDENTIFICATION = "llm_identification"
    HEURISTIC_FALLBACK = "heuristic_fallback"


@dataclass
class RecoveryResult:
    """Result of an element recovery attempt."""
    element: Optional[Element]
    strategy_used: RecoveryStrategy
    confidence_score: float
    recovery_time: float
    selector_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ElementSignature:
    """Signature of an element for caching and matching purposes."""
    tag_name: str
    attributes: Dict[str, str]
    text_content: str
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    visual_hash: str
    dom_path: str
    parent_context: str


class RecoveryCache:
    """Cache for successful recovery patterns."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[ElementSignature, str, float]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
    
    def _generate_cache_key(self, page_url: str, original_selector: str, context: str) -> str:
        """Generate a cache key from page URL, selector, and context."""
        key_data = f"{page_url}:{original_selector}:{context}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, page_url: str, original_selector: str, context: str = "") -> Optional[str]:
        """Retrieve a cached recovery pattern."""
        cache_key = self._generate_cache_key(page_url, original_selector, context)
        
        if cache_key in self.cache:
            signature, recovered_selector, timestamp = self.cache[cache_key]
            
            # Check if cache entry is still valid
            if time.time() - timestamp < self.ttl_seconds:
                self.hits += 1
                logger.debug(f"Cache hit for selector: {original_selector}")
                return recovered_selector
            else:
                # Remove expired entry
                del self.cache[cache_key]
        
        self.misses += 1
        return None
    
    def set(self, page_url: str, original_selector: str, 
            recovered_selector: str, signature: ElementSignature, context: str = ""):
        """Store a successful recovery pattern."""
        cache_key = self._generate_cache_key(page_url, original_selector, context)
        
        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k][2])
            del self.cache[oldest_key]
        
        self.cache[cache_key] = (signature, recovered_selector, time.time())
        logger.debug(f"Cached recovery pattern for selector: {original_selector}")
    
    def clear_expired(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, _, timestamp) in self.cache.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"Cleared {len(expired_keys)} expired cache entries")


class VisualSimilarityEngine:
    """Engine for visual similarity matching of elements."""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.feature_cache: Dict[str, np.ndarray] = {}
    
    async def find_visually_similar(self, page: Page, target_signature: ElementSignature,
                                   original_selector: str) -> Optional[Element]:
        """Find elements visually similar to the target signature."""
        try:
            # Get all interactive elements on the page
            all_elements = await self._get_interactive_elements(page)
            
            best_match = None
            best_score = 0.0
            
            for element in all_elements:
                try:
                    # Skip elements that match the original selector (already tried)
                    element_selector = await element.get_selector()
                    if element_selector == original_selector:
                        continue
                    
                    # Calculate visual similarity score
                    score = await self._calculate_visual_similarity(
                        page, element, target_signature
                    )
                    
                    if score > best_score and score >= self.similarity_threshold:
                        best_score = score
                        best_match = element
                        
                except Exception as e:
                    logger.debug(f"Error comparing element visually: {e}")
                    continue
            
            if best_match:
                logger.info(f"Found visually similar element with score: {best_score:.2f}")
                return best_match
            
        except Exception as e:
            logger.error(f"Visual similarity search failed: {e}")
        
        return None
    
    async def _get_interactive_elements(self, page: Page) -> List[Element]:
        """Get all interactive elements from the page including iframes and shadow DOM."""
        elements = []
        
        # Get elements from main frame
        main_elements = await page.query_selector_all(
            "button, input, select, textarea, a[href], [role='button'], "
            "[role='link'], [role='checkbox'], [role='radio'], [role='tab'], "
            "[onclick], [tabindex]"
        )
        elements.extend(main_elements)
        
        # Traverse iframes
        iframes = await page.query_selector_all("iframe")
        for iframe in iframes:
            try:
                frame = await iframe.content_frame()
                if frame:
                    frame_elements = await frame.query_selector_all(
                        "button, input, select, textarea, a[href], [role='button']"
                    )
                    elements.extend(frame_elements)
            except Exception as e:
                logger.debug(f"Failed to access iframe: {e}")
        
        # Traverse shadow DOMs
        shadow_hosts = await page.query_selector_all("*")
        for host in shadow_hosts:
            try:
                shadow_root = await host.evaluate("el => el.shadowRoot")
                if shadow_root:
                    shadow_elements = await host.evaluate(
                        """host => {
                            const elements = [];
                            const collect = (root) => {
                                const els = root.querySelectorAll(
                                    'button, input, select, textarea, a[href], [role="button"]'
                                );
                                elements.push(...els);
                                root.querySelectorAll('*').forEach(el => {
                                    if (el.shadowRoot) collect(el.shadowRoot);
                                });
                            };
                            collect(host.shadowRoot);
                            return elements;
                        }"""
                    )
                    # Convert shadow elements to Element objects
                    for shadow_el in shadow_elements:
                        # Create a temporary selector for the shadow element
                        temp_selector = await self._generate_selector_for_shadow_element(shadow_el)
                        element = await page.query_selector(temp_selector)
                        if element:
                            elements.append(element)
            except Exception as e:
                logger.debug(f"Failed to access shadow DOM: {e}")
        
        return elements
    
    async def _calculate_visual_similarity(self, page: Page, element: Element,
                                          target_signature: ElementSignature) -> float:
        """Calculate visual similarity score between element and target signature."""
        try:
            # Get element screenshot
            element_screenshot = await get_element_screenshot(page, element)
            if not element_screenshot:
                return 0.0
            
            # Convert to PIL Image
            img = Image.frombytes('RGB', 
                                 (element_screenshot['width'], element_screenshot['height']),
                                 element_screenshot['data'])
            
            # Calculate visual hash
            visual_hash = self._calculate_visual_hash(img)
            
            # Compare with target signature
            if visual_hash == target_signature.visual_hash:
                return 1.0
            
            # Calculate structural similarity
            structural_score = await self._calculate_structural_similarity(
                element, target_signature
            )
            
            # Calculate text similarity
            element_text = await element.text_content() or ""
            text_score = calculate_similarity_score(
                element_text, target_signature.text_content
            )
            
            # Weighted combination
            total_score = (
                0.4 * structural_score +
                0.3 * text_score +
                0.3 * (1.0 if visual_hash == target_signature.visual_hash else 0.0)
            )
            
            return total_score
            
        except Exception as e:
            logger.debug(f"Visual similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_visual_hash(self, image: Image.Image) -> str:
        """Calculate a perceptual hash of the image."""
        # Resize to 8x8 for perceptual hash
        resized = image.resize((8, 8), Image.Resampling.LANCZOS)
        grayscale = resized.convert('L')
        
        # Calculate average pixel value
        pixels = list(grayscale.getdata())
        avg = sum(pixels) / len(pixels)
        
        # Create hash based on pixels above/below average
        hash_bits = ''.join(['1' if pixel > avg else '0' for pixel in pixels])
        
        return hashlib.md5(hash_bits.encode()).hexdigest()
    
    async def _calculate_structural_similarity(self, element: Element,
                                              target_signature: ElementSignature) -> float:
        """Calculate structural similarity based on DOM properties."""
        try:
            element_tag = await element.evaluate("el => el.tagName.toLowerCase()")
            element_attrs = await element.evaluate(
                """el => {
                    const attrs = {};
                    for (const attr of el.attributes) {
                        attrs[attr.name] = attr.value;
                    }
                    return attrs;
                }"""
            )
            
            # Compare tag name
            tag_score = 1.0 if element_tag == target_signature.tag_name else 0.0
            
            # Compare attributes
            common_attrs = set(element_attrs.keys()) & set(target_signature.attributes.keys())
            if common_attrs:
                attr_matches = sum(
                    1 for attr in common_attrs 
                    if element_attrs[attr] == target_signature.attributes[attr]
                )
                attr_score = attr_matches / len(common_attrs)
            else:
                attr_score = 0.0
            
            # Compare bounding box size (not position, as layout may change)
            element_box = await element.bounding_box()
            if element_box and target_signature.bounding_box:
                target_width, target_height = target_signature.bounding_box[2], target_signature.bounding_box[3]
                width_ratio = min(element_box.width, target_width) / max(element_box.width, target_width)
                height_ratio = min(element_box.height, target_height) / max(element_box.height, target_height)
                size_score = (width_ratio + height_ratio) / 2
            else:
                size_score = 0.0
            
            # Weighted combination
            structural_score = (
                0.4 * tag_score +
                0.4 * attr_score +
                0.2 * size_score
            )
            
            return structural_score
            
        except Exception as e:
            logger.debug(f"Structural similarity calculation failed: {e}")
            return 0.0
    
    async def _generate_selector_for_shadow_element(self, shadow_element: Any) -> str:
        """Generate a CSS selector for an element inside shadow DOM."""
        # This is a simplified approach - in production, you'd need more sophisticated logic
        try:
            tag_name = await shadow_element.evaluate("el => el.tagName.toLowerCase()")
            id_attr = await shadow_element.evaluate("el => el.id")
            class_attr = await shadow_element.evaluate("el => el.className")
            
            if id_attr:
                return f"#{id_attr}"
            elif class_attr:
                classes = class_attr.split()
                return f"{tag_name}.{'.'.join(classes[:3])}"  # Limit to first 3 classes
            else:
                return tag_name
        except Exception:
            return "*"


class DOMSemanticAnalyzer:
    """Analyzer for semantic DOM patterns and relationships."""
    
    def __init__(self):
        self.semantic_patterns = self._load_semantic_patterns()
    
    def _load_semantic_patterns(self) -> Dict[str, List[str]]:
        """Load common semantic patterns for element identification."""
        return {
            'submit_button': [
                'button[type="submit"]',
                'input[type="submit"]',
                'button:contains("Submit")',
                'button:contains("Save")',
                '[role="button"][aria-label*="submit"]',
            ],
            'navigation_link': [
                'nav a',
                '[role="navigation"] a',
                'a[href^="/"]',
                'a[href^="http"]',
            ],
            'form_input': [
                'input:not([type="hidden"])',
                'textarea',
                'select',
                '[contenteditable="true"]',
            ],
            'close_button': [
                'button[aria-label*="close"]',
                '[role="button"][aria-label*="close"]',
                'button:contains("×")',
                'button:contains("Close")',
                '.close',
            ],
        }
    
    async def analyze_surrounding_dom(self, page: Page, original_selector: str,
                                     context: str = "") -> Optional[Element]:
        """Analyze surrounding DOM to find semantically similar elements."""
        try:
            # Parse the original selector to understand what we're looking for
            selector_info = parse_selector(original_selector)
            
            # Try different semantic strategies
            strategies = [
                self._find_by_text_similarity,
                self._find_by_attribute_patterns,
                self._find_by_dom_position,
                self._find_by_semantic_role,
            ]
            
            for strategy in strategies:
                element = await strategy(page, selector_info, context)
                if element:
                    return element
            
        except Exception as e:
            logger.error(f"DOM semantic analysis failed: {e}")
        
        return None
    
    async def _find_by_text_similarity(self, page: Page, selector_info: Dict,
                                      context: str) -> Optional[Element]:
        """Find elements with similar text content."""
        try:
            # Extract text from original selector if possible
            original_text = selector_info.get('text_content', '')
            if not original_text:
                return None
            
            # Find elements with similar text
            elements = await page.query_selector_all(f"text=/{original_text}/i")
            
            for element in elements:
                element_text = await element.text_content() or ""
                similarity = calculate_similarity_score(element_text, original_text)
                
                if similarity > 0.8:
                    return element
            
        except Exception as e:
            logger.debug(f"Text similarity search failed: {e}")
        
        return None
    
    async def _find_by_attribute_patterns(self, page: Page, selector_info: Dict,
                                         context: str) -> Optional[Element]:
        """Find elements with similar attribute patterns."""
        try:
            # Extract attributes from original selector
            original_attrs = selector_info.get('attributes', {})
            if not original_attrs:
                return None
            
            # Build attribute selectors
            attr_selectors = []
            for attr, value in original_attrs.items():
                if attr in ['id', 'class', 'name', 'type', 'role']:
                    attr_selectors.append(f'[{attr}*="{value}"]')
            
            if not attr_selectors:
                return None
            
            # Combine selectors with OR
            combined_selector = ', '.join(attr_selectors)
            elements = await page.query_selector_all(combined_selector)
            
            # Score each element
            best_match = None
            best_score = 0.0
            
            for element in elements:
                score = await self._calculate_attribute_similarity(element, original_attrs)
                if score > best_score:
                    best_score = score
                    best_match = element
            
            if best_score > 0.6:
                return best_match
            
        except Exception as e:
            logger.debug(f"Attribute pattern search failed: {e}")
        
        return None
    
    async def _find_by_dom_position(self, page: Page, selector_info: Dict,
                                   context: str) -> Optional[Element]:
        """Find elements based on DOM position relative to context."""
        try:
            if not context:
                return None
            
            # Find the context element first
            context_element = await page.query_selector(context)
            if not context_element:
                return None
            
            # Get siblings and children
            position_selectors = [
                f"{context} + *",  # Next sibling
                f"{context} ~ *",  # All following siblings
                f"{context} > *",  # Direct children
            ]
            
            for selector in position_selectors:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    # Check if element matches original selector pattern
                    if await self._matches_selector_pattern(element, selector_info):
                        return element
            
        except Exception as e:
            logger.debug(f"DOM position search failed: {e}")
        
        return None
    
    async def _find_by_semantic_role(self, page: Page, selector_info: Dict,
                                    context: str) -> Optional[Element]:
        """Find elements by ARIA roles and semantic HTML."""
        try:
            # Determine semantic role from selector
            tag_name = selector_info.get('tag_name', '')
            role = selector_info.get('attributes', {}).get('role', '')
            
            # Map to semantic patterns
            semantic_type = self._determine_semantic_type(tag_name, role, selector_info)
            
            if semantic_type in self.semantic_patterns:
                patterns = self.semantic_patterns[semantic_type]
                
                for pattern in patterns:
                    try:
                        element = await page.query_selector(pattern)
                        if element:
                            return element
                    except Exception:
                        continue
            
        except Exception as e:
            logger.debug(f"Semantic role search failed: {e}")
        
        return None
    
    async def _calculate_attribute_similarity(self, element: Element,
                                             original_attrs: Dict[str, str]) -> float:
        """Calculate similarity score based on attributes."""
        try:
            element_attrs = await element.evaluate(
                """el => {
                    const attrs = {};
                    for (const attr of el.attributes) {
                        attrs[attr.name] = attr.value;
                    }
                    return attrs;
                }"""
            )
            
            if not original_attrs:
                return 0.0
            
            matches = 0
            for attr, value in original_attrs.items():
                if attr in element_attrs:
                    element_value = element_attrs[attr]
                    # Check for exact match or substring match
                    if value == element_value or value in element_value:
                        matches += 1
            
            return matches / len(original_attrs)
            
        except Exception:
            return 0.0
    
    async def _matches_selector_pattern(self, element: Element,
                                       selector_info: Dict) -> bool:
        """Check if element matches the selector pattern."""
        try:
            tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
            if selector_info.get('tag_name') and tag_name != selector_info['tag_name']:
                return False
            
            # Check attributes
            for attr, value in selector_info.get('attributes', {}).items():
                element_value = await element.get_attribute(attr)
                if element_value != value:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _determine_semantic_type(self, tag_name: str, role: str,
                                selector_info: Dict) -> Optional[str]:
        """Determine semantic type from tag and role."""
        if tag_name == 'button' or role == 'button':
            text = selector_info.get('text_content', '').lower()
            if 'submit' in text or 'save' in text:
                return 'submit_button'
            elif 'close' in text or '×' in text:
                return 'close_button'
        
        elif tag_name == 'a' or role == 'link':
            return 'navigation_link'
        
        elif tag_name in ['input', 'textarea', 'select']:
            return 'form_input'
        
        return None


class LLMElementIdentifier:
    """LLM-powered element identification using screenshot context."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-vision-preview"):
        self.api_key = api_key
        self.model = model
        self.enabled = bool(api_key)
    
    async def identify_element(self, page: Page, original_selector: str,
                              context: str = "") -> Optional[Element]:
        """Identify element using LLM with screenshot context."""
        if not self.enabled:
            logger.warning("LLM identification disabled: no API key provided")
            return None
        
        try:
            # Take screenshot
            screenshot = await page.screenshot(type='jpeg', quality=70)
            
            # Get DOM context
            dom_context = await self._get_dom_context(page, original_selector, context)
            
            # Prepare prompt for LLM
            prompt = self._create_identification_prompt(original_selector, dom_context)
            
            # Call LLM API (simplified - in production, use proper API client)
            llm_response = await self._call_llm_api(prompt, screenshot)
            
            # Parse LLM response to get selector
            suggested_selector = self._parse_llm_response(llm_response)
            
            if suggested_selector:
                # Try the suggested selector
                element = await page.query_selector(suggested_selector)
                if element:
                    logger.info(f"LLM successfully identified element with selector: {suggested_selector}")
                    return element
            
        except Exception as e:
            logger.error(f"LLM identification failed: {e}")
        
        return None
    
    async def _get_dom_context(self, page: Page, original_selector: str,
                              context: str) -> str:
        """Get relevant DOM context for the LLM."""
        try:
            # Get a simplified DOM representation around the target area
            dom_snippet = await page.evaluate(
                """(selector, context) => {
                    // Try to find the context element
                    let contextEl = context ? document.querySelector(context) : null;
                    
                    // Get relevant part of DOM
                    let targetArea = contextEl || document.body;
                    
                    // Get a simplified representation
                    function simplifyDOM(el, depth = 0) {
                        if (depth > 3) return ''; // Limit depth
                        
                        const tag = el.tagName?.toLowerCase();
                        if (!tag) return '';
                        
                        const id = el.id ? `#${el.id}` : '';
                        const classes = el.className && typeof el.className === 'string' 
                            ? `.${el.className.split(' ').slice(0, 2).join('.')}` 
                            : '';
                        
                        const text = el.textContent?.trim().substring(0, 50) || '';
                        const children = Array.from(el.children || [])
                            .map(child => simplifyDOM(child, depth + 1))
                            .filter(Boolean)
                            .slice(0, 3); // Limit children
                        
                        const childStr = children.length > 0 
                            ? `\\n${children.join('\\n')}` 
                            : '';
                        
                        return `<${tag}${id}${classes}>${text ? ` "${text}"` : ''}${childStr}`;
                    }
                    
                    return simplifyDOM(targetArea);
                }""",
                original_selector,
                context
            )
            
            return dom_snippet or "No DOM context available"
            
        except Exception as e:
            logger.debug(f"Failed to get DOM context: {e}")
            return "Error retrieving DOM context"
    
    def _create_identification_prompt(self, original_selector: str,
                                     dom_context: str) -> str:
        """Create prompt for LLM element identification."""
        return f"""
        I need to find a web element that matches this selector: {original_selector}
        
        The original selector no longer works. Based on the screenshot and DOM context,
        please identify the element I'm looking for and provide a new CSS selector for it.
        
        DOM Context:
        {dom_context}
        
        Please respond with ONLY a CSS selector that would identify the correct element.
        If you cannot identify the element, respond with "NOT_FOUND".
        """
    
    async def _call_llm_api(self, prompt: str, screenshot: bytes) -> str:
        """Call LLM API with prompt and screenshot."""
        # This is a placeholder - in production, implement actual API call
        # For example, using OpenAI's API:
        """
        import openai
        import base64
        
        base64_image = base64.b64encode(screenshot).decode('utf-8')
        
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        return response.choices[0].message.content
        """
        
        # For now, return a mock response
        logger.warning("LLM API not implemented - using mock response")
        return "NOT_FOUND"
    
    def _parse_llm_response(self, response: str) -> Optional[str]:
        """Parse LLM response to extract selector."""
        response = response.strip()
        
        if response == "NOT_FOUND":
            return None
        
        # Basic validation of selector
        if any(invalid in response for invalid in ['<', '>', '{', '}', 'javascript:']):
            logger.warning(f"LLM returned potentially unsafe selector: {response}")
            return None
        
        return response


class ElementRecoverySystem:
    """Main element recovery system orchestrating all recovery strategies."""
    
    def __init__(self, page: Page, config: Optional[Dict[str, Any]] = None):
        self.page = page
        self.config = config or {}
        
        # Initialize components
        self.cache = RecoveryCache(
            max_size=self.config.get('cache_max_size', 1000),
            ttl_seconds=self.config.get('cache_ttl_seconds', 3600)
        )
        
        self.visual_engine = VisualSimilarityEngine(
            similarity_threshold=self.config.get('visual_similarity_threshold', 0.7)
        )
        
        self.dom_analyzer = DOMSemanticAnalyzer()
        
        self.llm_identifier = LLMElementIdentifier(
            api_key=self.config.get('llm_api_key'),
            model=self.config.get('llm_model', 'gpt-4-vision-preview')
        )
        
        # Recovery strategies in order
        self.strategies = [
            (RecoveryStrategy.ORIGINAL_SELECTOR, self._try_original_selector),
            (RecoveryStrategy.VISUAL_SIMILARITY, self._try_visual_similarity),
            (RecoveryStrategy.DOM_SEMANTICS, self._try_dom_semantics),
            (RecoveryStrategy.LLM_IDENTIFICATION, self._try_llm_identification),
            (RecoveryStrategy.HEURISTIC_FALLBACK, self._try_heuristic_fallback),
        ]
        
        # Statistics
        self.stats = {
            'total_attempts': 0,
            'successful_recoveries': 0,
            'strategy_successes': {strategy.value: 0 for strategy in RecoveryStrategy},
            'average_recovery_time': 0.0,
        }
    
    async def recover_element(self, original_selector: str,
                             context: str = "",
                             timeout: float = 10.0) -> RecoveryResult:
        """Attempt to recover an element using multiple strategies."""
        start_time = time.time()
        self.stats['total_attempts'] += 1
        
        logger.info(f"Starting element recovery for selector: {original_selector}")
        
        # Check cache first
        cached_selector = self.cache.get(
            self.page.url, original_selector, context
        )
        
        if cached_selector:
            try:
                element = await self.page.query_selector(cached_selector)
                if element:
                    recovery_time = time.time() - start_time
                    self._update_stats(RecoveryStrategy.ORIGINAL_SELECTOR, recovery_time, True)
                    
                    return RecoveryResult(
                        element=element,
                        strategy_used=RecoveryStrategy.ORIGINAL_SELECTOR,
                        confidence_score=0.95,  # High confidence for cached patterns
                        recovery_time=recovery_time,
                        selector_used=cached_selector,
                        metadata={'source': 'cache'}
                    )
            except Exception as e:
                logger.debug(f"Cached selector failed: {e}")
        
        # Try each strategy in order
        for strategy, strategy_func in self.strategies:
            try:
                logger.debug(f"Trying recovery strategy: {strategy.value}")
                
                element = await asyncio.wait_for(
                    strategy_func(original_selector, context),
                    timeout=timeout / len(self.strategies)
                )
                
                if element:
                    recovery_time = time.time() - start_time
                    
                    # Get the selector that worked
                    successful_selector = await element.get_selector()
                    
                    # Create element signature for caching
                    signature = await self._create_element_signature(element)
                    
                    # Cache the successful pattern
                    self.cache.set(
                        self.page.url, original_selector,
                        successful_selector, signature, context
                    )
                    
                    # Update statistics
                    self._update_stats(strategy, recovery_time, True)
                    
                    logger.info(f"Element recovered using {strategy.value} in {recovery_time:.2f}s")
                    
                    return RecoveryResult(
                        element=element,
                        strategy_used=strategy,
                        confidence_score=self._calculate_confidence(strategy, element),
                        recovery_time=recovery_time,
                        selector_used=successful_selector,
                        metadata={'signature': signature.__dict__}
                    )
                    
            except asyncio.TimeoutError:
                logger.debug(f"Strategy {strategy.value} timed out")
                continue
            except Exception as e:
                logger.debug(f"Strategy {strategy.value} failed: {e}")
                continue
        
        # All strategies failed
        recovery_time = time.time() - start_time
        self._update_stats(None, recovery_time, False)
        
        logger.warning(f"Failed to recover element: {original_selector}")
        
        return RecoveryResult(
            element=None,
            strategy_used=None,
            confidence_score=0.0,
            recovery_time=recovery_time,
            selector_used=original_selector,
            metadata={'error': 'All recovery strategies failed'}
        )
    
    async def _try_original_selector(self, original_selector: str,
                                    context: str) -> Optional[Element]:
        """Try the original selector as-is."""
        try:
            return await self.page.query_selector(original_selector)
        except Exception:
            return None
    
    async def _try_visual_similarity(self, original_selector: str,
                                    context: str) -> Optional[Element]:
        """Try visual similarity matching."""
        try:
            # Create a mock signature for the target element
            # In production, you'd want to store the original element's signature
            target_signature = ElementSignature(
                tag_name="",
                attributes={},
                text_content="",
                bounding_box=(0, 0, 0, 0),
                visual_hash="",
                dom_path="",
                parent_context=context
            )
            
            return await self.visual_engine.find_visually_similar(
                self.page, target_signature, original_selector
            )
        except Exception as e:
            logger.debug(f"Visual similarity failed: {e}")
            return None
    
    async def _try_dom_semantics(self, original_selector: str,
                                context: str) -> Optional[Element]:
        """Try DOM semantic analysis."""
        return await self.dom_analyzer.analyze_surrounding_dom(
            self.page, original_selector, context
        )
    
    async def _try_llm_identification(self, original_selector: str,
                                     context: str) -> Optional[Element]:
        """Try LLM-powered identification."""
        return await self.llm_identifier.identify_element(
            self.page, original_selector, context
        )
    
    async def _try_heuristic_fallback(self, original_selector: str,
                                     context: str) -> Optional[Element]:
        """Try heuristic fallback strategies."""
        try:
            # Parse the original selector
            selector_info = parse_selector(original_selector)
            
            # Try various heuristic strategies
            heuristics = [
                self._try_partial_attribute_match,
                self._try_text_content_match,
                self._try_positional_match,
                self._try_aria_label_match,
            ]
            
            for heuristic in heuristics:
                element = await heuristic(selector_info, context)
                if element:
                    return element
            
        except Exception as e:
            logger.debug(f"Heuristic fallback failed: {e}")
        
        return None
    
    async def _try_partial_attribute_match(self, selector_info: Dict,
                                          context: str) -> Optional[Element]:
        """Try matching elements with partial attribute values."""
        try:
            tag_name = selector_info.get('tag_name', '*')
            attributes = selector_info.get('attributes', {})
            
            if not attributes:
                return None
            
            # Build partial match selectors
            selectors = []
            for attr, value in attributes.items():
                if len(value) > 3:  # Only for non-trivial values
                    # Try contains, starts-with, ends-with
                    selectors.extend([
                        f'{tag_name}[{attr}*="{value}"]',
                        f'{tag_name}[{attr}^="{value[:len(value)//2]}"]',
                        f'{tag_name}[{attr}$="{value[len(value)//2:]}"]',
                    ])
            
            for selector in selectors:
                try:
                    element = await self.page.query_selector(selector)
                    if element:
                        return element
                except Exception:
                    continue
            
        except Exception as e:
            logger.debug(f"Partial attribute match failed: {e}")
        
        return None
    
    async def _try_text_content_match(self, selector_info: Dict,
                                     context: str) -> Optional[Element]:
        """Try matching by text content."""
        try:
            text_content = selector_info.get('text_content', '')
            if not text_content or len(text_content) < 3:
                return None
            
            # Use text selector
            selector = f'text=/{text_content}/i'
            return await self.page.query_selector(selector)
            
        except Exception as e:
            logger.debug(f"Text content match failed: {e}")
            return None
    
    async def _try_positional_match(self, selector_info: Dict,
                                   context: str) -> Optional[Element]:
        """Try matching by position relative to context."""
        try:
            if not context:
                return None
            
            # Get context element
            context_element = await self.page.query_selector(context)
            if not context_element:
                return None
            
            # Get context position
            context_box = await context_element.bounding_box()
            if not context_box:
                return None
            
            # Find elements near the context
            all_elements = await self.page.query_selector_all('*')
            
            for element in all_elements:
                try:
                    element_box = await element.bounding_box()
                    if not element_box:
                        continue
                    
                    # Check if element is near context
                    distance = self._calculate_distance(context_box, element_box)
                    if distance < 100:  # Within 100 pixels
                        # Check if element matches selector pattern
                        if await self._matches_selector_pattern(element, selector_info):
                            return element
                            
                except Exception:
                    continue
            
        except Exception as e:
            logger.debug(f"Positional match failed: {e}")
        
        return None
    
    async def _try_aria_label_match(self, selector_info: Dict,
                                   context: str) -> Optional[Element]:
        """Try matching by ARIA labels."""
        try:
            # Look for elements with ARIA labels
            aria_selectors = [
                '[aria-label]',
                '[aria-labelledby]',
                '[aria-describedby]',
                '[role]',
            ]
            
            for aria_selector in aria_selectors:
                elements = await self.page.query_selector_all(aria_selector)
                
                for element in elements:
                    try:
                        # Check if ARIA attributes match expected pattern
                        aria_label = await element.get_attribute('aria-label') or ''
                        role = await element.get_attribute('role') or ''
                        
                        # Simple heuristic: if role or label contains keywords from selector
                        selector_text = str(selector_info)
                        if any(keyword in (aria_label + role).lower() 
                               for keyword in ['button', 'link', 'input', 'submit']):
                            return element
                            
                    except Exception:
                        continue
            
        except Exception as e:
            logger.debug(f"ARIA label match failed: {e}")
        
        return None
    
    async def _create_element_signature(self, element: Element) -> ElementSignature:
        """Create a signature for an element."""
        try:
            tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
            
            attributes = await element.evaluate(
                """el => {
                    const attrs = {};
                    for (const attr of el.attributes) {
                        attrs[attr.name] = attr.value;
                    }
                    return attrs;
                }"""
            )
            
            text_content = await element.text_content() or ""
            
            bounding_box = await element.bounding_box()
            bbox_tuple = (
                bounding_box.x, bounding_box.y,
                bounding_box.width, bounding_box.height
            ) if bounding_box else (0, 0, 0, 0)
            
            # Get visual hash (simplified)
            visual_hash = hashlib.md5(text_content.encode()).hexdigest()
            
            # Get DOM path
            dom_path = await element.evaluate(
                """el => {
                    const path = [];
                    while (el && el.nodeType === Node.ELEMENT_NODE) {
                        let selector = el.tagName.toLowerCase();
                        if (el.id) {
                            selector += '#' + el.id;
                            path.unshift(selector);
                            break;
                        } else {
                            let sibling = el;
                            let nth = 1;
                            while (sibling.previousElementSibling) {
                                sibling = sibling.previousElementSibling;
                                nth++;
                            }
                            if (nth > 1) selector += ':nth-of-type(' + nth + ')';
                        }
                        path.unshift(selector);
                        el = el.parentElement;
                    }
                    return path.join(' > ');
                }"""
            )
            
            # Get parent context
            parent_context = await element.evaluate(
                """el => {
                    const parent = el.parentElement;
                    if (!parent) return '';
                    return parent.tagName.toLowerCase() + 
                           (parent.id ? '#' + parent.id : '') +
                           (parent.className && typeof parent.className === 'string' 
                            ? '.' + parent.className.split(' ').slice(0, 2).join('.') 
                            : '');
                }"""
            )
            
            return ElementSignature(
                tag_name=tag_name,
                attributes=attributes,
                text_content=text_content[:200],  # Limit length
                bounding_box=bbox_tuple,
                visual_hash=visual_hash,
                dom_path=dom_path,
                parent_context=parent_context
            )
            
        except Exception as e:
            logger.debug(f"Failed to create element signature: {e}")
            return ElementSignature(
                tag_name="",
                attributes={},
                text_content="",
                bounding_box=(0, 0, 0, 0),
                visual_hash="",
                dom_path="",
                parent_context=""
            )
    
    async def _matches_selector_pattern(self, element: Element,
                                       selector_info: Dict) -> bool:
        """Check if element matches the selector pattern."""
        try:
            tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
            if selector_info.get('tag_name') and tag_name != selector_info['tag_name']:
                return False
            
            # Check a few key attributes
            key_attrs = ['type', 'role', 'name']
            for attr in key_attrs:
                if attr in selector_info.get('attributes', {}):
                    element_value = await element.get_attribute(attr)
                    if element_value != selector_info['attributes'][attr]:
                        return False
            
            return True
            
        except Exception:
            return False
    
    def _calculate_distance(self, box1: Dict, box2: Dict) -> float:
        """Calculate distance between two bounding boxes."""
        center1_x = box1['x'] + box1['width'] / 2
        center1_y = box1['y'] + box1['height'] / 2
        center2_x = box2['x'] + box2['width'] / 2
        center2_y = box2['y'] + box2['height'] / 2
        
        return ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
    
    def _calculate_confidence(self, strategy: RecoveryStrategy,
                             element: Element) -> float:
        """Calculate confidence score for recovery."""
        base_scores = {
            RecoveryStrategy.ORIGINAL_SELECTOR: 1.0,
            RecoveryStrategy.VISUAL_SIMILARITY: 0.85,
            RecoveryStrategy.DOM_SEMANTICS: 0.75,
            RecoveryStrategy.LLM_IDENTIFICATION: 0.7,
            RecoveryStrategy.HEURISTIC_FALLBACK: 0.6,
        }
        
        return base_scores.get(strategy, 0.5)
    
    def _update_stats(self, strategy: Optional[RecoveryStrategy],
                     recovery_time: float, success: bool):
        """Update recovery statistics."""
        if success and strategy:
            self.stats['successful_recoveries'] += 1
            self.stats['strategy_successes'][strategy.value] += 1
        
        # Update average recovery time
        total_attempts = self.stats['total_attempts']
        current_avg = self.stats['average_recovery_time']
        self.stats['average_recovery_time'] = (
            (current_avg * (total_attempts - 1) + recovery_time) / total_attempts
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        return self.stats.copy()
    
    def clear_cache(self):
        """Clear the recovery cache."""
        self.cache.cache.clear()
        logger.info("Recovery cache cleared")


# Convenience function for easy integration
async def recover_element(page: Page, selector: str,
                         context: str = "",
                         config: Optional[Dict[str, Any]] = None) -> Optional[Element]:
    """
    Convenience function to recover an element.
    
    Args:
        page: The page to search in
        selector: The original CSS selector that failed
        context: Optional context selector to narrow search area
        config: Optional configuration dictionary
    
    Returns:
        The recovered element or None if recovery failed
    """
    recovery_system = ElementRecoverySystem(page, config)
    result = await recovery_system.recover_element(selector, context)
    return result.element


# Export main classes and functions
__all__ = [
    'ElementRecoverySystem',
    'RecoveryResult',
    'RecoveryStrategy',
    'RecoveryCache',
    'VisualSimilarityEngine',
    'DOMSemanticAnalyzer',
    'LLMElementIdentifier',
    'recover_element',
]