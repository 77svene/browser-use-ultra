import os
from typing import TYPE_CHECKING

from browser_use_ultra.logging_config import setup_logging

# Only set up logging if not in MCP mode or if explicitly requested
if os.environ.get('BROWSER_USE_SETUP_LOGGING', 'true').lower() != 'false':
	from browser_use_ultra.config import CONFIG

	# Get log file paths from config/environment
	debug_log_file = getattr(CONFIG, 'BROWSER_USE_DEBUG_LOG_FILE', None)
	info_log_file = getattr(CONFIG, 'BROWSER_USE_INFO_LOG_FILE', None)

	# Set up logging with file handlers if specified
	logger = setup_logging(debug_log_file=debug_log_file, info_log_file=info_log_file)
else:
	import logging

	logger = logging.getLogger('browser_use_ultra')

# Monkeypatch BaseSubprocessTransport.__del__ to handle closed event loops gracefully
from asyncio import base_subprocess

_original_del = base_subprocess.BaseSubprocessTransport.__del__


def _patched_del(self):
	"""Patched __del__ that handles closed event loops without throwing noisy red-herring errors like RuntimeError: Event loop is closed"""
	try:
		# Check if the event loop is closed before calling the original
		if hasattr(self, '_loop') and self._loop and self._loop.is_closed():
			# Event loop is closed, skip cleanup that requires the loop
			return
		_original_del(self)
	except RuntimeError as e:
		if 'Event loop is closed' in str(e):
			# Silently ignore this specific error
			pass
		else:
			raise


base_subprocess.BaseSubprocessTransport.__del__ = _patched_del


# Type stubs for lazy imports - fixes linter warnings
if TYPE_CHECKING:
	from browser_use_ultra.agent.prompts import SystemPrompt
	from browser_use_ultra.agent.service import Agent

	# from browser_use_ultra.agent.service import Agent
	from browser_use_ultra.agent.views import ActionModel, ActionResult, AgentHistoryList
	from browser_use_ultra.browser import BrowserProfile, BrowserSession
	from browser_use_ultra.browser import BrowserSession as Browser
	from browser_use_ultra.code_use.service import CodeAgent
	from browser_use_ultra.dom.service import DomService
	from browser_use_ultra.llm import models
	from browser_use_ultra.llm.anthropic.chat import ChatAnthropic
	from browser_use_ultra.llm.azure.chat import ChatAzureOpenAI
	from browser_use_ultra.llm.browser_use_ultra.chat import ChatBrowserUse
	from browser_use_ultra.llm.google.chat import ChatGoogle
	from browser_use_ultra.llm.groq.chat import ChatGroq
	from browser_use_ultra.llm.litellm.chat import ChatLiteLLM
	from browser_use_ultra.llm.mistral.chat import ChatMistral
	from browser_use_ultra.llm.oci_raw.chat import ChatOCIRaw
	from browser_use_ultra.llm.ollama.chat import ChatOllama
	from browser_use_ultra.llm.openai.chat import ChatOpenAI
	from browser_use_ultra.llm.vercel.chat import ChatVercel
	from browser_use_ultra.sandbox import sandbox
	from browser_use_ultra.tools.service import Controller, Tools

	# Lazy imports mapping - only import when actually accessed
_LAZY_IMPORTS = {
	# Agent service (heavy due to dependencies)
	# 'Agent': ('browser_use_ultra.agent.service', 'Agent'),
	# Code-use agent (Jupyter notebook-like execution)
	'CodeAgent': ('browser_use_ultra.code_use.service', 'CodeAgent'),
	'Agent': ('browser_use_ultra.agent.service', 'Agent'),
	# System prompt (moderate weight due to agent.views imports)
	'SystemPrompt': ('browser_use_ultra.agent.prompts', 'SystemPrompt'),
	# Agent views (very heavy - over 1 second!)
	'ActionModel': ('browser_use_ultra.agent.views', 'ActionModel'),
	'ActionResult': ('browser_use_ultra.agent.views', 'ActionResult'),
	'AgentHistoryList': ('browser_use_ultra.agent.views', 'AgentHistoryList'),
	'BrowserSession': ('browser_use_ultra.browser', 'BrowserSession'),
	'Browser': ('browser_use_ultra.browser', 'BrowserSession'),  # Alias for BrowserSession
	'BrowserProfile': ('browser_use_ultra.browser', 'BrowserProfile'),
	# Tools (moderate weight)
	'Tools': ('browser_use_ultra.tools.service', 'Tools'),
	'Controller': ('browser_use_ultra.tools.service', 'Controller'),  # alias
	# DOM service (moderate weight)
	'DomService': ('browser_use_ultra.dom.service', 'DomService'),
	# Chat models (very heavy imports)
	'ChatOpenAI': ('browser_use_ultra.llm.openai.chat', 'ChatOpenAI'),
	'ChatGoogle': ('browser_use_ultra.llm.google.chat', 'ChatGoogle'),
	'ChatAnthropic': ('browser_use_ultra.llm.anthropic.chat', 'ChatAnthropic'),
	'ChatBrowserUse': ('browser_use_ultra.llm.browser_use_ultra.chat', 'ChatBrowserUse'),
	'ChatGroq': ('browser_use_ultra.llm.groq.chat', 'ChatGroq'),
	'ChatLiteLLM': ('browser_use_ultra.llm.litellm.chat', 'ChatLiteLLM'),
	'ChatMistral': ('browser_use_ultra.llm.mistral.chat', 'ChatMistral'),
	'ChatAzureOpenAI': ('browser_use_ultra.llm.azure.chat', 'ChatAzureOpenAI'),
	'ChatOCIRaw': ('browser_use_ultra.llm.oci_raw.chat', 'ChatOCIRaw'),
	'ChatOllama': ('browser_use_ultra.llm.ollama.chat', 'ChatOllama'),
	'ChatVercel': ('browser_use_ultra.llm.vercel.chat', 'ChatVercel'),
	# LLM models module
	'models': ('browser_use_ultra.llm.models', None),
	# Sandbox execution
	'sandbox': ('browser_use_ultra.sandbox', 'sandbox'),
	# Resilient network layer components
	'CircuitBreaker': ('browser_use_ultra.browser.resilient', 'CircuitBreaker'),
	'ResilientCDPClient': ('browser_use_ultra.browser.resilient', 'ResilientCDPClient'),
	'NetworkHealthMonitor': ('browser_use_ultra.browser.resilient', 'NetworkHealthMonitor'),
}


def __getattr__(name: str):
	"""Lazy import mechanism - only import modules when they're actually accessed."""
	if name in _LAZY_IMPORTS:
		module_path, attr_name = _LAZY_IMPORTS[name]
		try:
			from importlib import import_module

			module = import_module(module_path)
			if attr_name is None:
				# For modules like 'models', return the module itself
				attr = module
			else:
				attr = getattr(module, attr_name)
			# Cache the imported attribute in the module's globals
			globals()[name] = attr
			return attr
		except ImportError as e:
			raise ImportError(f'Failed to import {name} from {module_path}: {e}') from e

	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
	'Agent',
	'CodeAgent',
	# 'CodeAgent',
	'BrowserSession',
	'Browser',  # Alias for BrowserSession
	'BrowserProfile',
	'Controller',
	'DomService',
	'SystemPrompt',
	'ActionResult',
	'ActionModel',
	'AgentHistoryList',
	# Chat models
	'ChatOpenAI',
	'ChatGoogle',
	'ChatAnthropic',
	'ChatBrowserUse',
	'ChatGroq',
	'ChatLiteLLM',
	'ChatMistral',
	'ChatAzureOpenAI',
	'ChatOCIRaw',
	'ChatOllama',
	'ChatVercel',
	'Tools',
	'Controller',
	# LLM models module
	'models',
	# Sandbox execution
	'sandbox',
	# Resilient network layer
	'CircuitBreaker',
	'ResilientCDPClient',
	'NetworkHealthMonitor',
]