#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/5 22:59
@Author  : alexanderwu
@File    : __init__.py
"""
import importlib
class LLMFactory:
    def __init__(self, module_name, instance_name):
        self.module_name = module_name
        self.instance_name = instance_name
        self._module = None

    def __getattr__(self, name):
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
        return getattr(self._module, name)
    def __instancecheck__(self, instance):
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
        return isinstance(instance, getattr(self._module, self.instance_name))

    
GeminiLLM = LLMFactory("metagpt.provider.google_gemini_api ", "GeminiLLM")
OllamaLLM = LLMFactory("metagpt.provider.ollama_api ", "OllamaLLM")
OpenAILLM = LLMFactory("metagpt.provider.openai_api ", "OpenAILLM")
ZhiPuAILLM = LLMFactory("metagpt.provider.zhipuai_api ", "ZhiPuAILLM")
AzureOpenAILLM = LLMFactory("metagpt.provider.azure_openai_api ", "AzureOpenAILLM")
MetaGPTLLM = LLMFactory("metagpt.provider.metagpt_api ", "MetaGPTLLM")
HumanProvider = LLMFactory("metagpt.provider.human_provider ", "HumanProvider")
SparkLLM = LLMFactory("metagpt.provider.spark_api ", "SparkLLM")
QianFanLLM = LLMFactory("metagpt.provider.qianfan_api ", "QianFanLLM")
DashScopeLLM = LLMFactory("metagpt.provider.dashscope_api ", "DashScopeLLM")
AnthropicLLM = LLMFactory("metagpt.provider.anthropic_api ", "AnthropicLLM")
BedrockLLM = LLMFactory("metagpt.provider.bedrock_api ", "BedrockLLM")
ArkLLM = LLMFactory("metagpt.provider.ark_api ", "ArkLLM")

__all__ = [
    "GeminiLLM",
    "OpenAILLM",
    "ZhiPuAILLM",
    "AzureOpenAILLM",
    "MetaGPTLLM",
    "OllamaLLM",
    "HumanProvider",
    "SparkLLM",
    "QianFanLLM",
    "DashScopeLLM",
    "AnthropicLLM",
    "BedrockLLM",
    "ArkLLM",
]
