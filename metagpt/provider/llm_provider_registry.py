#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/19 17:26
@Author  : alexanderwu
@File    : llm_provider_registry.py
"""
from metagpt.configs.llm_config import LLMConfig, LLMType
from metagpt.provider.base_llm import BaseLLM
import importlib

class LLMProviderRegistry:
    def __init__(self):
        self.providers = {}
        self._module_map = {
            LLMType.OPENAI: "metagpt.provider.openai_api",
            LLMType.ANTHROPIC: "metagpt.provider.anthropic_api",
            LLMType.CLAUDE: "metagpt.provider.anthropic_api",  # Same module as Anthropic
            LLMType.SPARK: "metagpt.provider.spark_api",
            LLMType.ZHIPUAI: "metagpt.provider.zhipuai_api",
            LLMType.FIREWORKS: "metagpt.provider.fireworks_api",
            LLMType.OPEN_LLM: "metagpt.provider.open_llm_api",
            LLMType.GEMINI: "metagpt.provider.google_gemini_api",
            LLMType.METAGPT: "metagpt.provider.metagpt_api",
            LLMType.AZURE: "metagpt.provider.azure_openai_api",
            LLMType.OLLAMA: "metagpt.provider.ollama_api",
            LLMType.QIANFAN: "metagpt.provider.qianfan_api",  # Baidu BCE
            LLMType.DASHSCOPE: "metagpt.provider.dashscope_api",  # Aliyun LingJi DashScope
            LLMType.MOONSHOT: "metagpt.provider.moonshot_api",
            LLMType.MISTRAL: "metagpt.provider.mistral_api",
            LLMType.YI: "metagpt.provider.yi_api",  # lingyiwanwu
            LLMType.OPENROUTER: "metagpt.provider.openrouter_api",
            LLMType.BEDROCK: "metagpt.provider.bedrock_api",
            LLMType.ARK: "metagpt.provider.ark_api",
        }

    def register(self, key, provider_cls):
        self.providers[key] = provider_cls

    def get_provider(self, enum: LLMType):
        """get provider instance according to the enum"""
        if enum not in self.providers:
            # Import and register the provider if not already registered
            module_name = self._module_map[enum]
            importlib.import_module(module_name)
        return self.providers[enum]


def register_provider(keys):
    """register provider to registry"""

    def decorator(cls):
        if isinstance(keys, list):
            for key in keys:
                LLM_REGISTRY.register(key, cls)
        else:
            LLM_REGISTRY.register(keys, cls)
        return cls

    return decorator


def create_llm_instance(config: LLMConfig) -> BaseLLM:
    """get the default llm provider"""
    llm = LLM_REGISTRY.get_provider(config.api_type)(config)
    if llm.use_system_prompt and not config.use_system_prompt:
        # for models like o1-series, default openai provider.use_system_prompt is True, but it should be False for o1-*
        llm.use_system_prompt = config.use_system_prompt
    return llm


# Registry instance
LLM_REGISTRY = LLMProviderRegistry()
