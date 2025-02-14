from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import azure.cognitiveservices.speech as speechsdk
import os
import requests
import importlib
from itertools import permutations
import base64
import random
import pandas as pd
import json
from pathvalidate import sanitize_filename
from PIL import Image
import fitz
import ast
import re
import wave
from charset_normalizer import detect
import codecs
import math
from decimal import Decimal, getcontext
from scraper import get_web_texts, get_web_contents
from request_web_search import tavily_answer, tavily_news_by_freshness, bing_by_freshness, bing_by_year_range, google_answer, google_by_freshness, google_by_year_range, google_scholar_by_year_range, google_patents_by_year_range, duckduckgo_by_freshness, duckduckgo_by_year_range, exa_by_freshness, exa_by_year_range, exa_news_by_freshness, exa_paper_by_freshness, exa_paper_by_year_range
from export_to_word import export_search_results_to_word, append_company_info_and_disclaimer
from aife_time import now_and_choices, iso_date, get_recent_dates_iso
from aife_utils import retrieve, manage_thread, upload_to_container, upload_and_send

OPENROUTER_API_KEY = retrieve("OpenRouter")
RAINBOW_API_KEY = retrieve("Rainbow")
HYPERBOLIC_API_KEY = retrieve("Hyperbolic")
DEEPINFRA_API_KEY = retrieve("DeepInfra")
DASHSCOPE_API_KEY = retrieve("DashScope")
SILICONFLOW_API_KEY = retrieve("SiliconFlow")
LUCHEN_API_KEY = retrieve("Luchen")
LUCHEN2_API_KEY = retrieve("Luchen2")
DEEPSEEK_API_KEY = retrieve("DeepSeek")
YI_API_KEY = retrieve("Lingyiwanwu")
MINIMAX_API_KEY = retrieve("MiniMax")
XAI_API_KEY = retrieve("xAI")
EXCELLENCE_API_KEY = retrieve("ExcellenceKey")
EXCELLENCE_ENDPOINT = retrieve("ExcellenceEndpoint")
EXCELLENCE2_API_KEY = retrieve("Excellence2Key")
EXCELLENCE2_ENDPOINT = retrieve("Excellence2Endpoint")

DOCUMENT_CONFIGS = [{"api_key": retrieve("YusiMultiKey"), "endpoint": retrieve("YusiMultiEndpoint")},
                   {"api_key": retrieve("YusiMulti2Key"), "endpoint": retrieve("YusiMulti2Endpoint")}]

SPEECH_CONFIGS = [{"api_key": retrieve("SpeechKey"), "region": "eastus"},
                  {"api_key": retrieve("Speech2Key"), "region": "westus2"}]


def execute(tool_calls):
    try:
        requests = []
        for tool_call in tool_calls:
            if function := tool_call.get("function"):
                if (name := function.get("name")) and (arguments := function.get("arguments")):
                    requests.append((globals()[name], json.loads(arguments)))
        return {f"{function.__name__}({arguments})": result for result, function, arguments in manage_thread(requests)}
    except Exception as e:
        print(f"Failed to execute tool calls: {e}")
        return None


def request_llm(url, headers, data):
    for attempt in range(2):
        try:
            print(f"Sending request to {url} with data: {data}")
            response = requests.post(url, headers=headers, json=data, timeout=120).json()
            print(response)
            if message := response.get("choices", [{}])[0].get("message", {}):
                if tool_calls := message.get("tool_calls"):
                    if tool_results := execute(tool_calls):
                        return {"type": "yusi_tool_result", "tool_results": tool_results}
                elif content := message.get("content"):
                    if reasoning_content := message.get("reasoning_content"):
                        return f"<think>\n{reasoning_content}\n</think>\n\n{content}"
                    elif citations := response.get("citations"):
                        return f"{content}\n\n{'\n'.join(f'{i + 1}. {citation}' for i, citation in enumerate(citations))}"
                    else:
                        return content
            raise Exception(f"Invalid response: {response} or execution failed")
        except Exception as e:
            print(f"Request LLM attempt {attempt + 1} failed: {e}")
    return None


class LLM:
    def __init__(self, url, api_key):
        self.url = url
        self.api_key = api_key

    def __call__(self, messages, model, temperature=None, top_p=None, max_tokens=None, response_format=None, tools=None):
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "messages": messages,
            "model": model,
            **({"temperature": temperature} if temperature is not None else {}),
            **({"top_p": top_p} if top_p else {}),
            **({"max_tokens": max_tokens} if max_tokens else {}),
            **({"response_format": response_format} if response_format else {}),
            **({"tools": tools} if tools else {})
        }
        return request_llm(self.url, headers, data)


class AzureOpenAI:
    def __init__(self, endpoint, api_key):
        self.endpoint = endpoint
        self.api_key = api_key

    def __call__(self, messages, model, temperature=None, top_p=None, max_tokens=None, response_format=None, tools=None):
        url = f"{self.endpoint}openai/deployments/{model}/chat/completions?api-version=2024-10-21"
        headers = {
            "api-key": self.api_key
        }
        data = {
            "messages": messages,
            **({"temperature": temperature} if temperature is not None else {}),
            **({"top_p": top_p} if top_p else {}),
            **({"max_tokens": max_tokens} if max_tokens else {}),
            **({"response_format": response_format} if response_format else {}),
            **({"tools": tools} if tools else {})
        }
        return request_llm(url, headers, data)


openrouter = LLM("https://openrouter.ai/api/v1/chat/completions", OPENROUTER_API_KEY)
rainbow = LLM("https://gitaigc.com/v1/chat/completions", RAINBOW_API_KEY)
hyperbolic = LLM("https://api.hyperbolic.xyz/v1/chat/completions", HYPERBOLIC_API_KEY)
deepinfra = LLM("https://api.deepinfra.com/v1/openai/chat/completions", DEEPINFRA_API_KEY)
dashscope = LLM("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", DASHSCOPE_API_KEY)
siliconflow = LLM("https://api.siliconflow.cn/v1/chat/completions", SILICONFLOW_API_KEY)
luchen = LLM("https://cloud.luchentech.com/api/maas/chat/completions", LUCHEN_API_KEY)
luchen2 = LLM("https://cloud.luchentech.com/api/maas/chat/completions", LUCHEN2_API_KEY)
deepseek = LLM("https://api.deepseek.com/chat/completions", DEEPSEEK_API_KEY)
lingyiwanwu = LLM("https://api.lingyiwanwu.com/v1/chat/completions", YI_API_KEY)
minimax = LLM("https://api.minimax.chat/v1/text/chatcompletion_v2", MINIMAX_API_KEY)
xai = LLM("https://api.x.ai/v1/chat/completions", XAI_API_KEY)
excellence = AzureOpenAI(EXCELLENCE_ENDPOINT, EXCELLENCE_API_KEY)
excellence2 = AzureOpenAI(EXCELLENCE2_ENDPOINT, EXCELLENCE2_API_KEY)


def get_prompt(name, **arguments):
    prompt = getattr(importlib.import_module(f"aife_prompts.{name}"), name)
    if arguments:
        if callable(prompt):
            return prompt(**arguments)
        else:
            return prompt.format(**arguments)
    else:
        if callable(prompt):
            return prompt()
        else:
            return prompt


def get_response_format(response_format):
    if response_format:
        return getattr(importlib.import_module(f"aife_response_formats.{response_format}"), response_format)
    return None


def get_tools(tools):
    if tools:
        return [getattr(importlib.import_module("aife_tools"), tool) for tool in tools]
    return None


class YusiChat:
    def __call__(self, configs, messages, response_format=None, tools=None):
        for config in configs:
            try:
                llm_config = llm_configs[config]
                result = globals()[llm_config["name"]](messages, **llm_config["arguments"], response_format=response_format, tools=tools)
                if result:
                    return result
            except Exception:
                continue
        return None


yusi_chat = YusiChat()


def get_all_permutations(input_list):
    return {i: list(permutation) for i, permutation in enumerate(permutations(input_list))}


def text_chat(configs, system_message, user_message, response_format=None, tools=None, index=None):
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]
    return yusi_chat(configs, messages, response_format, tools)


def text_chats(configs, system_message, user_messages, response_format=None, tools=None):
    configs = get_all_permutations(configs)
    requests = [(text_chat, (configs[i % len(configs)], system_message, user_message, response_format, tools, i)) for i, user_message in enumerate(user_messages if isinstance(user_messages, list) else [user_messages], start=0) if user_message]
    return {arguments[-1]: result for result, function, arguments in manage_thread(requests)}


def image_chat(configs, user_message, image_path):
    messages = [{"role": "user", "content": [{"type": "text", "text": user_message}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')}"}}]}]
    return yusi_chat(configs, messages)


def image_chats(configs, user_message, image_paths):
    configs = get_all_permutations(configs)
    requests = [(image_chat, (configs[i % len(configs)], user_message, image_path)) for i, image_path in enumerate(image_paths if isinstance(image_paths, list) else [image_paths], start=0) if image_path]
    return {arguments[-1]: result for result, function, arguments in manage_thread(requests)}


def request_stt(url, headers, files, data):
    for attempt in range(2):
        try:
            print(f"Sending request to {url}")
            response = requests.post(url, headers=headers, files=files, data=data, timeout=120).json()
            if text := response.get("text"):
                return text
            raise Exception(f"Invalid response: {response}")
        except Exception as e:
            print(f"Request STT attempt {attempt + 1} failed: {e}")
    return None


def speech_to_text(file_path):
    url = "https://api.deepinfra.com/v1/openai/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}"
    }
    files = {
        "file": open(file_path, "rb")
    }
    data = {
        "model": "openai/whisper-large-v3-turbo"
    }
    return request_stt(url, headers, files, data)


jobs = {
    "Text chat": {
        "category": "text",
        "llms": ["deepseek_chat", "deepseek_reasoner", "qwen_max", "phi4", "grok2", "o3_mini_high", "gpt4o", "claude35_sonnet", "gemini2_flash", "minimax_01", "yi_lightning", "lfm_7b"],
        "system_message": "chat_only",
        "response_format": None,
        "tools": None
    },
    "Multimodal chat": {
        "category": "multimodal",
        "llms": ["gpt4o", "claude35_sonnet", "gemini2_flash", "minimax_01"],
        "system_message": "chat_only",
        "response_format": None,
        "tools": None
    },
    "Web search (built-in)": {
        "category": "text",
        "llms": ["perplexity"],
        "system_message": "perplexity_chat_and_search",
        "response_format": None,
        "tools": None
    },
    "Web search (function calling)": {
        "category": "multimodal",
        "llms": ["gpt4o", "claude35_sonnet"],
        "system_message": "chat_and_search",
        "response_format": None,
        "tools": ["basic_search_func", "serious_search_by_freshness_func", "serious_search_by_year_range_func", "news_search_by_freshness_func", "scholar_search_by_freshness_func", "scholar_search_by_year_range_func", "patents_search_by_year_range_func", "get_web_texts_func"]
    },
    "Read documents": {
        "category": "multimodal",
        "llms": ["gpt4o", "claude35_sonnet"],
        "system_message": "interpret_documents_with_tools",
        "response_format": None,
        "tools": ["generate_audio_interpretation_func"]
    },
    "Read long documents": {
        "category": "multimodal",
        "llms": ["gemini2_flash", "minimax_01"],
        "system_message": "interpret_documents",
        "response_format": None,
        "tools": None
    },
    "Math and coding": {
        "category": "multimodal",
        "llms": ["gpt4o", "claude35_sonnet"],
        "system_message": "math_and_coding",
        "response_format": None,
        "tools": ["calculator_func"]
    },
    "Critical thinking": {
        "category": "text",
        "llms": ["deepseek_chat", "qwen_max", "phi4", "grok2", "o3_mini_high", "gpt4o", "claude35_sonnet", "gemini2_flash", "minimax_01", "yi_lightning", "lfm_7b"],
        "system_message": "logic_checking_and_critical_thinking",
        "response_format": None,
        "tools": None
    },
    "Translation and optimisation": {
        "category": "text",
        "llms": ["deepseek_chat", "qwen_max", "phi4", "grok2", "o3_mini_high", "gpt4o", "claude35_sonnet", "gemini2_flash", "minimax_01", "yi_lightning", "lfm_7b"],
        "system_message": "language_translation_and_optimisation",
        "response_format": None,
        "tools": None
    }
}

llms = {
    "deepseek_chat": {
        "configs": ["deepseek_chat_hyperbolic", "deepseek_chat", "deepseek_chat_openrouter"],
        "intro": "DeepSeek-V3",
        "context_length": 65536
    },
    "deepseek_reasoner": {
        "configs": ["deepseek_reasoner_hyperbolic", "deepseek_reasoner_dashscope", "deepseek_reasoner_luchen", "deepseek_reasoner_luchen2", "deepseek_reasoner"],
        "intro": "DeepSeek-R1",
        "context_length": 65536
    },
    "perplexity": {
        "configs": ["perplexity_openrouter"],
        "intro": "DeepSeek R1结合perplexity联网搜索",
        "context_length": 127000
    },
    "qwen_max": {
        "configs": ["qwen_max_openrouter"],
        "intro": "阿里云通义千问旗舰模型",
        "context_length": 32768
    },
    "grok2": {
        "configs": ["grok2_xai", "grok2_openrouter"],
        "intro": "马斯克xAI全球最大超算集群训练的模型",
        "context_length": 131072
    },
    "o3_mini_high": {
        "configs": ["o3_mini_high_openrouter"],
        "intro": "OpenAI o3-mini reasoning_effort设为高",
        "context_length": 200000
    },
    "gpt4o": {
        "configs": ["gpt4o_rainbow", "gpt4o_openrouter", "gpt4o_excellence"],
        "intro": "OpenAI GPT-4o",
        "context_length": 128000
    },
    "claude35_sonnet": {
        "configs": ["claude35_sonnet_openrouter"],
        "intro": "Claude 3.5 Sonnet",
        "context_length": 200000
    },
    "gemini2_flash": {
        "configs": ["gemini2_flash_openrouter"],
        "intro": "Google Gemini系列最新多模态模型",
        "context_length": 1000192
    },
    "minimax_01": {
        "configs": ["minimax_01", "minimax_01_openrouter"],
        "intro": "MiniMax-01系列模型",
        "context_length": 1000192
    },
    "phi4": {
        "configs": ["phi4_deepinfra", "phi4_openrouter"],
        "intro": "微软Phi系列精选数据训练的模型",
        "context_length": 16384
    },
    "yi_lightning": {
        "configs": ["yi_lightning"],
        "intro": "李开复零一万物旗舰模型",
        "context_length": 16384
    },
    "lfm_7b": {
        "configs": ["lfm_7b_openrouter"],
        "intro": "Liquid AI非Transformer架构模型",
        "context_length": 32768
    },
    "gpt4o_mini": {
        "configs": ["gpt4o_mini_rainbow", "gpt4o_mini_openrouter", "gpt4o_mini_excellence"],
        "intro": None,
        "context_length": 128000
    },
    "qwen_vl_plus": {
        "configs": ["qwen2_vl_7b_openrouter", "qwen2_vl_7b_siliconflow", "qwen_vl_plus_dashscope"],
        "intro": None,
        "context_length": None
    },
    "qwen_vl_max": {
        "configs": ["qwen2_vl_72b_openrouter", "qwen2_vl_72b_siliconflow", "qwen_vl_max_dashscope"],
        "intro": None,
        "context_length": None
    }
}

llm_configs = {
    "deepseek_chat_hyperbolic": {
        "name": "hyperbolic",
        "arguments": {
            "model": "deepseek-ai/DeepSeek-V3",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "deepseek_chat": {
        "name": "deepseek",
        "arguments": {
            "model": "deepseek-chat",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "deepseek_chat_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "deepseek/deepseek-chat",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "deepseek_reasoner_dashscope": {
        "name": "dashscope",
        "arguments": {
            "model": "deepseek-r1",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": None
        }
    },
    "deepseek_reasoner_hyperbolic": {
        "name": "hyperbolic",
        "arguments": {
            "model": "deepseek-ai/DeepSeek-R1",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": None
        }
    },
    "deepseek_reasoner_luchen": {
        "name": "luchen",
        "arguments": {
            "model": "deepseek_r1",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": None
        }
    },
    "deepseek_reasoner_luchen2": {
        "name": "luchen2",
        "arguments": {
            "model": "deepseek_r1",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": None
        }
    },
    "deepseek_reasoner": {
        "name": "deepseek",
        "arguments": {
            "model": "deepseek-reasoner",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": None
        }
    },
    "perplexity_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "perplexity/sonar-reasoning",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "qwen_max_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "qwen/qwen-max",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "grok2_xai": {
        "name": "xai",
        "arguments": {
            "model": "grok-2-1212",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "grok2_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "x-ai/grok-2-1212",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "o3_mini_high_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "openai/o3-mini-high",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": None
        }
    },
    "gpt4o_rainbow": {
        "name": "rainbow",
        "arguments": {
            "model": "gpt-4o-2024-08-06",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "gpt4o_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "openai/gpt-4o-2024-08-06",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "gpt4o_excellence": {
        "name": "excellence",
        "arguments": {
            "model": "excellence",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "claude35_sonnet_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "anthropic/claude-3.5-sonnet",
            "temperature": 0.1,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "gemini2_flash_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "google/gemini-2.0-flash-001",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "minimax_01": {
        "name": "minimax",
        "arguments": {
            "model": "MiniMax-Text-01",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "minimax_01_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "minimax/minimax-01",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "phi4_deepinfra": {
        "name": "deepinfra",
        "arguments": {
            "model": "microsoft/phi-4",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "phi4_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "microsoft/phi-4",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "yi_lightning": {
        "name": "lingyiwanwu",
        "arguments": {
            "model": "yi-lightning",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "lfm_7b_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "liquid/lfm-7b",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "gpt4o_mini_rainbow": {
        "name": "rainbow",
        "arguments": {
            "model": "gpt-4o-mini-2024-07-18",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "gpt4o_mini_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "openai/gpt-4o-mini-2024-07-18",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "gpt4o_mini_excellence": {
        "name": "excellence2",
        "arguments": {
            "model": "yusi-mini",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "qwen_vl_plus_dashscope": {
        "name": "dashscope",
        "arguments": {
            "model": "qwen-vl-plus-2025-01-02",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": None
        }
    },
    "qwen2_vl_7b_siliconflow": {
        "name": "siliconflow",
        "arguments": {
            "model": "Pro/Qwen/Qwen2-VL-7B-Instruct",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": None
        }
    },
    "qwen2_vl_7b_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "qwen/qwen-2-vl-7b-instruct",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": None
        }
    },
    "qwen_vl_max_dashscope": {
        "name": "dashscope",
        "arguments": {
            "model": "qwen-vl-max-2024-12-30",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": None
        }
    },
    "qwen2_vl_72b_siliconflow": {
        "name": "siliconflow",
        "arguments": {
            "model": "Qwen/Qwen2-VL-72B-Instruct",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": None
        }
    },
    "qwen2_vl_72b_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "qwen/qwen-2-vl-72b-instruct",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": None
        }
    }
}

basic_search_func = {
    "type": "function",
    "function": {
        "name": "basic_search",
        "description": "Utilise search engines to obtain the answer box, knowledge graph, and/or a list of webpages. Call this when explicit answers are required for an expected reply.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The input to search engines, which can be keywords or phrases, and may also include advanced search operators."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
}

news_search_by_freshness_func = {
    "type": "function",
    "function": {
        "name": "news_search_by_freshness",
        "description": "Utilise search engines to access news articles, optionally filtering by the number of days ago. Call this when details of public occurrences, particularly recent occurrences, are required for an expected reply, such as news on climate change from the past week.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_target": {
                    "type": "string",
                    "description": "A specification of the objects or answers to search for. This will be used for multiple rounds of screening and filtering to optimise the search results."
                },
                "query": {
                    "type": "string",
                    "description": "The input to search engines, which can be keywords or phrases."
                },
                "freshness": {
                    "type": ["integer", "null"],
                    "description": "The number of days to look back for filtering the results by published date. The common values are 7 or 30. Omit this if no specific recency is required."
                }
            },
            "required": ["search_target", "query"],
            "additionalProperties": False
        }
    }
}

scholar_search_by_freshness_func = {
    "type": "function",
    "function": {
        "name": "scholar_search_by_freshness",
        "description": "Utilise search engines to access research articles, optionally filtering by the number of days ago. Call this when scholarly content, particularly recent content, is required for an expected reply, such as papers on AI ethics from the past month.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_target": {
                    "type": "string",
                    "description": "A specification of the objects or answers to search for. This will be used for multiple rounds of screening and filtering to optimise the search results."
                },
                "query": {
                    "type": "string",
                    "description": "The input to search engines, which can be keywords or phrases."
                },
                "freshness": {
                    "type": ["integer", "null"],
                    "description": "The number of days to look back for filtering the results by published date. The common values are 7 or 30. Omit this if no specific recency is required."
                }
            },
            "required": ["search_target", "query"],
            "additionalProperties": False
        }
    }
}

serious_search_by_freshness_func = {
    "type": "function",
    "function": {
        "name": "serious_search_by_freshness",
        "description": "Utilise search engines to access information in specified areas of expertise, optionally filtering by the number of days ago. Call this when professional content, particularly recent content, is required for an expected reply, such as the product releases of humanoid robots by European companies this month.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_target": {
                    "type": "string",
                    "description": "A specification of the objects or answers to search for. This will be used for multiple rounds of screening and filtering to optimise the search results."
                },
                "query": {
                    "type": "string",
                    "description": "The input to search engines, which can be keywords or phrases, and may also include advanced search operators."
                },
                "freshness": {
                    "type": ["integer", "null"],
                    "description": "The number of days to look back for filtering the results by published date. The common values are 7 or 30. Omit this if no specific recency is required."
                }
            },
            "required": ["search_target", "query"],
            "additionalProperties": False
        }
    }
}

scholar_search_by_year_range_func = {
    "type": "function",
    "function": {
        "name": "scholar_search_by_year_range",
        "description": "Utilise search engines to access research articles, optionally filtering by year range. Call this when scholarly content, particularly in specific years, is required for an expected reply, such as papers on precision medicine published between 2022 and 2024.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_target": {
                    "type": "string",
                    "description": "A specification of the objects or answers to search for. This will be used for multiple rounds of screening and filtering to optimise the search results."
                },
                "query": {
                    "type": "string",
                    "description": "The input to search engines, which can be keywords or phrases."
                },
                "year_range": {
                    "type": ["array", "null"],
                    "description": "The start year and end year for filtering the results by published date. This is a pair of 4-digit integers, in which the first represents the start year and the second represents the end year. Example: [2022, 2024]. Omit this if no specific year range is required.",
                    "items": {
                        "type": "integer",
                        "description": "A 4-digit integer representing a year."
                    },
                    "minItems": 2,
                    "maxItems": 2
                }
            },
            "required": ["search_target", "query"],
            "additionalProperties": False
        }
    }
}

patents_search_by_year_range_func = {
    "type": "function",
    "function": {
        "name": "patents_search_by_year_range",
        "description": "Utilise Google Search to access patent files, optionally filtering by year range. Call this when patent information, particularly in specific years, is required for an expected reply, such as patents on electric vehicles granted in the last three years.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_target": {
                    "type": "string",
                    "description": "A specification of the objects or answers to search for. This will be used for multiple rounds of screening and filtering to optimise the search results."
                },
                "query": {
                    "type": "string",
                    "description": "The input to search engines, which can be keywords or phrases."
                },
                "year_range": {
                    "type": ["array", "null"],
                    "description": "The start year and end year for filtering the results by granted date. This is a pair of 4-digit integers, in which the first represents the start year and the second represents the end year. Example: [2022, 2024]. Omit this if no specific year range is required.",
                    "items": {
                        "type": "integer",
                        "description": "A 4-digit integer representing a year."
                    },
                    "minItems": 2,
                    "maxItems": 2
                }
            },
            "required": ["search_target", "query"],
            "additionalProperties": False
        }
    }
}

serious_search_by_year_range_func = {
    "type": "function",
    "function": {
        "name": "serious_search_by_year_range",
        "description": "Utilise search engines to access information in specified areas of expertise, optionally filtering by year range. Call this when professional content, particularly in specific years, is required for an expected reply, such as the world's top tech companies by market capitalisation over the past 3 years.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_target": {
                    "type": "string",
                    "description": "A specification of the objects or answers to search for. This will be used for multiple rounds of screening and filtering to optimise the search results."
                },
                "query": {
                    "type": "string",
                    "description": "The input to search engines, which can be keywords or phrases, and may also include advanced search operators."
                },
                "year_range": {
                    "type": ["array", "null"],
                    "description": "The start year and end year for filtering the results by published date. This is a pair of 4-digit integers, in which the first represents the start year and the second represents the end year. Example: [2022, 2024]. Omit this if no specific year range is required.",
                    "items": {
                        "type": "integer",
                        "description": "A 4-digit integer representing a year."
                    },
                    "minItems": 2,
                    "maxItems": 2
                }
            },
            "required": ["search_target", "query"],
            "additionalProperties": False
        }
    }
}

get_web_texts_func = {
    "type": "function",
    "function": {
        "name": "get_web_texts",
        "description": "Scrape the full text of each webpage from the provided URLs. Call this when there are single or multiple URLs and the user requests or you need to browse one or more of the webpages.",
        "parameters": {
            "type": "object",
            "properties": {
                "web_urls": {
                    "type": "array",
                    "description": "A list of URLs to simultaneously scrape text from.",
                    "items": {
                        "type": "string",
                        "description": "The URL to scrape text from."
                    }
                }
            },
            "required": ["web_urls"],
            "additionalProperties": False
        }
    }
}

generate_audio_interpretation_func = {
    "type": "function",
    "function": {
        "name": "generate_audio_interpretation",
        "description": "Generate an audio interpretation for reports, plans, and studies. This function returns a URL for downloading the audio file.",
        "parameters": {
            "type": "object",
            "properties": {
                "txt_path": {
                    "type": "string",
                    "description": "The path of the TXT file containing the entire content of the documents."
                },
                "user_requirements": {
                    "type": "string",
                    "description": "The user's requirements for the interpretation."
                },
                "voice_gender": {
                    "type": "string",
                    "enum": ["male", "female"],
                    "description": "The user's choice of male or female voice for the audio."
                },
                "to_email": {
                    "type": ["string", "null"],
                    "description": "The user's email address for secondary delivery. This is optional, as a safeguard against potential chat session disruptions."
                },
            },
            "required": ["txt_path", "user_requirements", "voice_gender"],
            "additionalProperties": False,
        },
    },
}

calculator_func = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a wide range of mathematical expressions with high precision (50 decimal places). Supports basic arithmetic operations and math module functions.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate. Examples: '2*3.14159', 'Decimal(1)/Decimal(3)', 'math.sqrt(2)', 'math.sin(math.pi/2)'",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    }
}


def search_results_to_csv(search_target, results, freshness=None):
    csv_path = f"temp-data/{sanitize_filename(search_target[:20])} {now_and_choices()}.csv"
    pd.DataFrame(columns=["title", "summary", "publication_info", "url", "date", "grant_date"]).to_csv(csv_path, index=False, encoding="utf-8")
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = pd.concat([df, pd.DataFrame([{**{column: result.get(column) for column in df.columns if column in result}} for result in results]).reindex(columns=df.columns)])
    df = df.drop_duplicates(subset="title", keep="first")
    if freshness:
        df = df.drop([i for i, row in df.iterrows() if row.get("date") not in get_recent_dates_iso(freshness)])
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


def relevant_search_results_to_csv(csv_path, results):
    pd.DataFrame(columns=["title", "web_text", "publication_info", "url", "date", "grant_date"]).to_csv(csv_path, index=False, encoding="utf-8")
    df = pd.read_csv(csv_path, encoding="utf-8")
    web_texts = get_web_texts([result["url"] for i, result in results.items()])
    df = pd.concat([df, pd.DataFrame([{"web_text": web_texts[result["url"]], **{column: result.get(column) for column in df.columns if column in result}} for i, result in results.items() if web_texts[result["url"]]]).reindex(columns=df.columns)])
    df.to_csv(csv_path, index=False, encoding="utf-8")


def screen_search_results(csv_path, search_target):
    system_message = get_prompt("are_relevant_search_results", search_target=search_target)
    response_format = get_response_format("are_relevant_search_results_json")
    df = pd.read_csv(csv_path, encoding="utf-8")
    chunks = [df.iloc[i:i + 30] for i in range(0, len(df), 30)]
    user_messages = [json.dumps(chunk[["title", "summary"]].to_dict("index"), ensure_ascii=False) for chunk in chunks]
    for attempt in range(3):
        try:
            results = {i: json.loads(result) for i, result in text_chats(["gpt4o_mini_rainbow", "gpt4o_mini_openrouter", "gpt4o_mini_excellence"], system_message, user_messages, response_format).items()}
            if relevant_search_results := {key: value for i, result in results.items() for key, value in chunks[i].loc[chunks[i].index.isin(result["indices"])].to_dict("index").items()}:
                relevant_search_results_to_csv(csv_path, relevant_search_results)
                return True
            return None
        except Exception:
            return None


def matched_search_results_to_csv(csv_path, results):
    pd.DataFrame(columns=["title", "key_points", "publication_info", "url", "date", "grant_date"]).to_csv(csv_path, index=False, encoding="utf-8")
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = pd.concat([df, pd.DataFrame([{**{column: result.get(column) for column in df.columns if column in result}} for i, result in results.items()]).reindex(columns=df.columns)])
    df.to_csv(csv_path, index=False, encoding="utf-8")


def filter_search_results(csv_path, search_target):
    system_message = get_prompt("is_matched_search_result", search_target=search_target)
    response_format = get_response_format("is_matched_search_result_json")
    records = pd.read_csv(csv_path, encoding="utf-8").to_dict("records")
    user_messages = [record["web_text"] for record in records]
    for attempt in range(3):
        try:
            results = {i: json.loads(result) for i, result in text_chats(["gpt4o_mini_rainbow", "gpt4o_mini_openrouter", "gpt4o_mini_excellence"], system_message, user_messages, response_format).items()}
            if matched_search_results := {i: {**records[i], "key_points": f"{result['key_points']}\n\n{result['analysis']}"} for i, result in results.items() if result["judgement"]}:
                matched_search_results_to_csv(csv_path, matched_search_results)
                return True
            return None
        except Exception:
            return None


def gather_search_results(csv_path, existing_results=None):
    df = pd.read_csv(csv_path, encoding="utf-8")
    columns_to_check = ["title", "key_points", "publication_info", "url", "date", "grant_date"]
    results = [{column: row[column] for column in columns_to_check if pd.notna(row.get(column))} for i, row in df.iterrows()]
    if not existing_results:
        return json.dumps(results, ensure_ascii=False, indent=4)
    else:
        existing_results.extend(results)
        return json.dumps(existing_results, ensure_ascii=False, indent=4)


def serious_search(search_target, query, freshness_or_year_range, system_message, response_format, retry=0):
    if isinstance(freshness_or_year_range, list) and len(freshness_or_year_range) == 2:
        results = serious_search_by_year_range(search_target, query, freshness_or_year_range)
    elif isinstance(freshness_or_year_range, int) or freshness_or_year_range is None:
        results = serious_search_by_freshness(search_target, query, freshness_or_year_range)
    else:
        results = None
    if results:
        print(f"Serious search results: {results}")
        user_message = f"The query is:\n{query}\n\nThe existing_results are:\n{json.dumps(results, ensure_ascii=False, indent=4)}"
        for attempt in range(3):
            try:
                result = json.loads(text_chat(["gpt4o_rainbow", "gpt4o_openrouter", "gpt4o_excellence"], system_message, user_message, response_format))
                if result["judgement"]:
                    return results
                elif new_query := result["new_query"] and retry < 5:
                    return serious_search(search_target, new_query, freshness_or_year_range, system_message, response_format, retry + 1)
            except Exception:
                return None
    return None


def retrieved_results_to_csv(retrieved_results):
    csv_path = f"temp-data/{now_and_choices()}.csv"
    pd.DataFrame(columns=["web_url", "web_raw_content", "heading_1", "heading_2", "source", "published_date", "web_content", "body_content"]).to_csv(csv_path, index=False, encoding="utf-8")
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = pd.concat([df, pd.DataFrame([{"heading_1": heading_1, "web_url": web_url}
        for heading_1, web_urls in retrieved_results.items()
        for web_url in web_urls]).reindex(columns=df.columns)])
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


def web_contents_from_url_to_csv(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8")
    valid_mask = df["web_url"].notna()
    web_urls = df[valid_mask]["web_url"].tolist()
    web_contents = get_web_contents(web_urls)
    df.loc[valid_mask, "web_content"] = df.loc[valid_mask, "web_url"].map(web_contents)
    valid_mask = df["web_content"].notna()
    df = df[valid_mask]
    df.to_csv(csv_path, index=False, encoding="utf-8")


def extend_body_content_bounds(web_content, body_content_bounds):
    start_bound, end_bound = body_content_bounds
    while start_bound - 1 in web_content and web_content[start_bound - 1].startswith("temp-images"):
        start_bound -= 1
    while end_bound + 1 in web_content and web_content[end_bound + 1].startswith("temp-images"):
        end_bound += 1
    return start_bound, end_bound


def extract_info_from_online_article(web_url, web_content):
    system_message = get_prompt("extract_info_from_online_article")
    response_format = get_response_format("extract_info_from_online_article_json")
    user_message = f"<web_content>{(dict(list(web_content.items())[:80] + list(web_content.items())[-80:]) if len(web_content) > 160 else web_content)}</web_content>"
    for attempt in range(3):
        try:
            results = json.loads(text_chat(["gpt4o_rainbow", "gpt4o_openrouter", "gpt4o_excellence"], system_message, user_message, response_format))
            title = results.get("title")
            source = results.get("source")
            published_date = iso_date(results.get("published_date"))
            body_content_bounds = results.get("body_content_bounds")
            if title and source and published_date and len(body_content_bounds) == 2:
                start_bound, end_bound = extend_body_content_bounds(web_content, body_content_bounds)
                body_content = {key: web_content[key] for key in range(start_bound, end_bound + 1) if key in web_content}
                return title, source, published_date, body_content
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    print("Failed to extract info after maximum retries")
    return None, None, None, None


def extract_info_from_online_articles(web_urls, web_contents):
    requests = [(extract_info_from_online_article, (web_url, web_content)) for web_url, web_content in zip(web_urls, web_contents)]
    return {arguments[0]: result for result, name, arguments in manage_thread(requests)}


def info_from_web_contents_to_csv(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8")
    valid_mask = df["web_content"].notna()
    web_urls = df[valid_mask]["web_url"].tolist()
    web_contents = [ast.literal_eval(web_content) for web_content in df[valid_mask]["web_content"].tolist()]
    info = extract_info_from_online_articles(web_urls, web_contents)
    df.loc[valid_mask, "heading_2"] = df.loc[valid_mask, "web_url"].map({web_url: values[0] for web_url, values in info.items()})
    df.loc[valid_mask, "source"] = df.loc[valid_mask, "web_url"].map({web_url: values[1] for web_url, values in info.items()})
    df.loc[valid_mask, "published_date"] = df.loc[valid_mask, "web_url"].map({web_url: values[2] for web_url, values in info.items()})
    df.loc[valid_mask, "body_content"] = df.loc[valid_mask, "web_url"].map({web_url: str(values[3]) if values[3] else None for web_url, values in info.items()})
    df.to_csv(csv_path, index=False, encoding="utf-8")


def online_articles_from_url_to_word(retrieved_results):
    csv_path = retrieved_results_to_csv(retrieved_results)
    web_contents_from_url_to_csv(csv_path)
    info_from_web_contents_to_csv(csv_path)
    doc_path = export_search_results_to_word(csv_path)
    append_company_info_and_disclaimer(doc_path)
    return upload_to_container(doc_path)


def information_retrieval(search_strategies):
    response_format = get_response_format("are_adequate_search_results_json")
    requests = []
    for search_strategy in search_strategies:
        if (search_target := search_strategy.get("search_target")) and (query := search_strategy.get("query")):
            freshness_or_year_range = search_strategy.get("freshness_or_year_range")
            system_message = get_prompt("are_adequate_search_results", search_target=search_target)
            requests.append((serious_search, (search_target, query, freshness_or_year_range, system_message, response_format)))
    retrieved_results = {arguments[0]: [item["url"] for item in json.loads(result) if "url" in item] for result, function, arguments in manage_thread(requests) if result}
    print(f"Retrieved results: {retrieved_results}")
    return online_articles_from_url_to_word(retrieved_results)


def basic_search(query):
    requests = [
        (tavily_answer, (query,)),
        (google_answer, (query,))
    ]
    if results := {key: value for result, function, arguments in manage_thread(requests) if result for key, value in result.items()}:
        return results
    return None


def news_search_by_freshness(search_target, query, freshness=None):
    freshness = freshness if isinstance(freshness, int) else None
    requests = [
        (tavily_news_by_freshness, (query, freshness)),
        (google_by_freshness, (query, freshness)),
        (exa_news_by_freshness, (query, freshness))
    ]
    if results := [item for result, function, arguments in manage_thread(requests) if result for item in result]:
        csv_path = search_results_to_csv(search_target, results, freshness)
        if screen_search_results(csv_path, search_target):
            if filter_search_results(csv_path, search_target):
                return gather_search_results(csv_path)
    return None


def scholar_search_by_freshness(search_target, query, freshness=None):
    freshness = freshness if isinstance(freshness, int) else None
    requests = [
        (exa_paper_by_freshness, (query, freshness))
    ]
    if results := [item for result, function, arguments in manage_thread(requests) if result for item in result]:
        csv_path = search_results_to_csv(search_target, results, freshness)
        if screen_search_results(csv_path, search_target):
            if filter_search_results(csv_path, search_target):
                return gather_search_results(csv_path)
    return None


def serious_search_by_freshness(search_target, query, freshness=None):
    freshness = freshness if isinstance(freshness, int) else None
    requests = [
        (bing_by_freshness, (query, freshness)),
        (google_by_freshness, (query, freshness)),
        (duckduckgo_by_freshness, (query, freshness)),
        (exa_by_freshness, (query, freshness))
    ]
    if results := [item for result, function, arguments in manage_thread(requests) if result for item in result]:
        csv_path = search_results_to_csv(search_target, results, freshness)
        if screen_search_results(csv_path, search_target):
            if filter_search_results(csv_path, search_target):
                return gather_search_results(csv_path)
    return None


def scholar_search_by_year_range(search_target, query, year_range=None):
    year_range = year_range if len(year_range) == 2 and all(isinstance(x, int) for x in year_range) else None
    requests = [
        (google_scholar_by_year_range, (query, year_range)),
        (exa_paper_by_year_range, (query, year_range))
    ]
    if results := [item for result, function, arguments in manage_thread(requests) if result for item in result]:
        print(results)
        csv_path = search_results_to_csv(search_target, results)
        if screen_search_results(csv_path, search_target):
            if filter_search_results(csv_path, search_target):
                return gather_search_results(csv_path)
    return None


def patents_search_by_year_range(search_target, query, year_range=None):
    year_range = year_range if len(year_range) == 2 and all(isinstance(x, int) for x in year_range) else None
    requests = [
        (google_patents_by_year_range, (query, year_range))
    ]
    if results := [item for result, function, arguments in manage_thread(requests) if result for item in result]:
        csv_path = search_results_to_csv(search_target, results)
        if screen_search_results(csv_path, search_target):
            if filter_search_results(csv_path, search_target):
                return gather_search_results(csv_path)
    return None


def serious_search_by_year_range(search_target, query, year_range=None):
    year_range = year_range if len(year_range) == 2 and all(isinstance(x, int) for x in year_range) else None
    requests = [
        (bing_by_year_range, (query, year_range)),
        (google_by_year_range, (query, year_range)),
        (duckduckgo_by_year_range, (query, year_range)),
        (exa_by_year_range, (query, year_range))
    ]
    if results := [item for result, function, arguments in manage_thread(requests) if result for item in result]:
        csv_path = search_results_to_csv(search_target, results)
        if screen_search_results(csv_path, search_target):
            if filter_search_results(csv_path, search_target):
                return gather_search_results(csv_path)
    return None


def resize_image(image_path, side_limit):
    with Image.open(image_path) as f:
        if max(f.size) > side_limit:
            ratio = side_limit / max(f.size)
            resized = f.resize((int(f.size[0] * ratio), int(f.size[1] * ratio)), Image.Resampling.LANCZOS)
            with open(image_path, "wb") as out_f:
                resized.save(out_f, f.format)


def resize_images(image_paths, side_limit):
    requests = []
    for image_path in image_paths:
        if image_path:
            request = (resize_image, (image_path, side_limit))
            requests.append(request)
    manage_thread(requests)


def is_plain_text(pdf_path, side_limit=1280):
    image_paths = []
    with fitz.open(pdf_path) as f:
        for i in range(f.page_count):
            image_path = f"temp-images/{now_and_choices()}.png"
            f.load_page(i).get_pixmap().save(image_path)
            if os.path.isfile(image_path):
                image_paths.append(image_path)
            else:
                image_paths.append(None)
    if image_paths:
        resize_images(image_paths, side_limit)
        user_message = get_prompt("is_plain_text")
        results = image_chats(["qwen2_vl_7b_openrouter", "qwen2_vl_7b_siliconflow", "qwen_vl_plus_dashscope"], user_message, image_paths)
        return [image_path if image_path and results.get(image_path) == "False" else None for image_path in image_paths]
    return None


def describe_visual_elements(pdf_path):
    image_paths = is_plain_text(pdf_path)
    if image_paths and any(image_paths):
        user_message = get_prompt("describe_visual_elements")
        results = image_chats(["qwen2_vl_72b_openrouter", "qwen2_vl_72b_siliconflow", "qwen_vl_max_dashscope"], user_message, image_paths)
        return [results.get(image_path) if image_path else None for image_path in image_paths]
    return None


def extract_page_texts(pdf_path, pages_per_chunk, overlap_length=100):
    for attempt, config in enumerate(random.sample(DOCUMENT_CONFIGS, len(DOCUMENT_CONFIGS)), 1):
        try:
            client = DocumentAnalysisClient(endpoint=config["endpoint"], credential=AzureKeyCredential(config["api_key"]))
            with open(pdf_path, "rb") as f:
                page_texts = ["\n".join([line.content for line in page.lines]) for page in client.begin_analyze_document("prebuilt-read", f).result().pages]
                for i in range(len(page_texts)):
                    if i > 0 and i % pages_per_chunk == 0:
                        last_chunk = page_texts[i - 1]
                        overlap_text = last_chunk[-overlap_length:] if len(last_chunk) > overlap_length else ""
                        page_texts[i] = overlap_text + " " + page_texts[i]
                return page_texts
        except Exception:
            continue
    try:
        with fitz.open(pdf_path) as f:
            page_texts = [page.get_text() for page in f]
            for i in range(len(page_texts)):
                if i > 0 and i % pages_per_chunk == 0:
                    last_chunk = page_texts[i - 1]
                    overlap_text = last_chunk[-overlap_length:] if len(last_chunk) > overlap_length else ""
                    page_texts[i] = overlap_text + " " + page_texts[i]
            return page_texts
    except Exception as e:
        print(f"Failed to extract page texts: {e}")
        return None


def parse_pdfs(pdf_paths, pages_per_chunk, overlap_length, is_plain_text=False):
    contents = {}
    for pdf_path in pdf_paths:
        if is_plain_text:
            page_texts = extract_page_texts(pdf_path, pages_per_chunk, overlap_length)
            if page_texts:
                pdf_contents = {i // pages_per_chunk + 1: "\n\n".join(page_text for page_text in page_texts[i:i + pages_per_chunk]) for i in range(0, len(page_texts), pages_per_chunk)}
                contents.update({(len(contents) + key): value for key, value in pdf_contents.items()})
        else:
            requests = [
                (describe_visual_elements, (pdf_path,)),
                (extract_page_texts, (pdf_path, pages_per_chunk, overlap_length))
            ]
            results = {function.__name__: result for result, function, arguments in manage_thread(requests)}
            if page_texts := results.get("extract_page_texts"):
                if page_descriptions := results.get("describe_visual_elements"):
                    pdf_contents = {i // pages_per_chunk + 1: "\n\n".join((page_descriptions[i + j] + "\n\n" if i + j < len(page_descriptions) and page_descriptions[i + j] else "") + page_text for j, page_text in enumerate(page_texts[i:i + pages_per_chunk])) for i in range(0, len(page_texts), pages_per_chunk)}
                else:
                    pdf_contents = {i // pages_per_chunk + 1: "\n\n".join(page_text for page_text in page_texts[i:i + pages_per_chunk]) for i in range(0, len(page_texts), pages_per_chunk)}
                contents.update({(len(contents) + key): value for key, value in pdf_contents.items()})
    return contents


def parse_txts(txt_paths, chars_per_chunk, overlap_length):
    contents = {}
    for txt_path in txt_paths:
        with open(txt_path, "r", encoding="utf-8") as f:
            if content := f.read():
                chunks = [content[i:i + chars_per_chunk] for i in range(0, len(content), chars_per_chunk)]
                for i in range(1, len(chunks)):
                    last_chunk = chunks[i - 1]
                    overlap_text = last_chunk[-overlap_length:] if len(last_chunk) > overlap_length else ""
                    chunks[i] = overlap_text + " " + chunks[i]
                txt_contents = {i + 1: chunk for i, chunk in enumerate(chunks)}
                contents.update({(len(contents) + key): value for key, value in txt_contents.items()})
    return contents


def ensure_csv_utf8(file_path):
    try:
        if file_path.endswith(".csv"):
            with open(file_path, "rb") as f:
                encoding = codecs.lookup(detect(f.read(min(32768, os.path.getsize(file_path))))["encoding"]).name
            df = pd.read_csv(file_path, encoding=encoding)
        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path, engine="openpyxl")
        else:
            print(f"Unsupported file format: {file_path}")
            return None
        first_valid_column = next((column for column in df.columns if pd.notna(column) or df[column].notna().any()), None)
        if first_valid_column:
            empty_mask = df[first_valid_column].isna()
            if empty_mask.any():
                df.loc[empty_mask, first_valid_column] = df.index[empty_mask]
            csv_path = os.path.splitext(file_path)[0] + ".csv"
            df.to_csv(csv_path, index=False, encoding="utf-8")
            return csv_path
    except Exception as e:
        print(f"Error in ensure_csv_utf8: {e}")
    return None


def parse_csvs(csv_paths):
    contents = {}
    for i, csv_path in enumerate(csv_paths):
        records = pd.read_csv(csv_path, encoding="utf-8").to_dict("records")
        contents[i] = records
    return json.dumps(contents, indent=4, ensure_ascii=False)


def process_csv_content(file_path, column_to_process, column_of_results, system_message):
    csv_path = ensure_csv_utf8(file_path)
    df = pd.read_csv(csv_path, encoding="utf-8")
    df[column_of_results] = None
    valid_mask = df[column_to_process].notna()
    contents = df.loc[valid_mask, column_to_process].tolist()
    user_messages = [f"<{column_to_process}>\n{content}\n</{column_to_process}>" for content in contents]
    results = text_chats(["gpt4o_mini_rainbow", "gpt4o_mini_openrouter", "gpt4o_mini_excellence"], system_message, user_messages)
    df.loc[valid_mask, column_of_results] = [results[i] for i in range(1, len(contents) + 1)]
    csv_path = csv_path.replace(".csv", "_processed.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


def read_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return ast.literal_eval(f.read())


def extract_text_with_xml_tag(text, tag):
    if matched := re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL):
        return matched.group(1).strip()
    else:
        return re.sub(r"<[^>]+>", "", text).strip()


def parse_text_with_pattern(text, pattern):
    if markers := [(marker.start(), marker.group()) for marker in re.finditer(pattern, text)]:
        return {i + 1: text[marker[0] + len(marker[1]):markers[i + 1][0] if i < len(markers) - 1 else len(text)].strip() for i, marker in enumerate(markers)}
    else:
        return None


def filter_words(text, words):
    for word in words:
        text = text.replace(word, "")
    return text


def azure_tts(text, voice_gender):
    wav_path = f"temp-data/{now_and_choices()}.wav"
    for attempt, config in enumerate(random.sample(SPEECH_CONFIGS, len(SPEECH_CONFIGS)), 1):
        try:
            speech_config = speechsdk.SpeechConfig(config["api_key"], config["region"])
            speech_config.speech_synthesis_voice_name = "zh-CN-YunxiaoMultilingualNeural" if voice_gender == "male" else "zh-CN-XiaoyuMultilingualNeural"
            synthesizer = speechsdk.SpeechSynthesizer(speech_config, speechsdk.AudioConfig(filename=wav_path))
            if synthesizer.speak_text_async(text).get():
                return wav_path
        except Exception as e:
            print(f"Azure TTS attempt {attempt} failed: {e}")
    return None


def generate_audios_serially(texts, voice_gender):
    return {key: azure_tts(text, voice_gender) for key, text in texts.items()}


def merge_audios(audios):
    valid_audios = []
    for key in sorted(audios):
        try:
            with wave.open(audios[key], "rb") as f:
                if not valid_audios:
                    params = f.getparams()
                valid_audios.append(audios[key])
        except Exception:
            continue
    if valid_audios:
        wav_path = f"temp-data/{now_and_choices()}.wav"
        with wave.open(wav_path, "wb") as out_f:
            out_f.setparams(params)
            for valid_audio in valid_audios:
                with wave.open(valid_audio, "rb") as in_f:
                    out_f.writeframes(in_f.readframes(in_f.getnframes()))
        return wav_path


def outline_for_audio_interpretation(csv_path, user_requirements, doc_content):
    system_message = get_prompt("outline_for_audio_interpretation")
    user_message = f"<user_requirements>\n{user_requirements}\n</user_requirements>\n<doc_content>\n{doc_content}\n</doc_content>"
    outline = extract_text_with_xml_tag(text_chat(["minimax_01", "minimax_01_openrouter"], system_message, user_message), "outline")
    headings_and_notes = parse_text_with_pattern(outline, r"Dialogue Question of Chapter \d+:|Notes:")
    if isinstance(headings_and_notes, dict):
        *headings, notes = headings_and_notes.items()
        df = pd.DataFrame(columns=list(dict(headings).values()))
        if os.path.isfile(csv_path):
            df = pd.concat([pd.read_csv(csv_path, encoding="utf-8"), df], axis=1)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        return outline, None if notes[1] == "None" else notes[1]
    return None


def information_for_audio_interpretation(doc_contents, outline):
    system_message = get_prompt("information_for_audio_interpretation")
    user_messages = [f"<outline>\n{outline}\n</outline>\n<doc_content>\n{doc_content}\n</doc_content>" for doc_content in doc_contents.values()]
    results = text_chats(["gpt4o_mini_rainbow", "gpt4o_mini_openrouter", "gpt4o_mini_excellence"], system_message, user_messages)
    return {i: parse_text_with_pattern(extract_text_with_xml_tag(results[i], "information_for_questions"), r"Information for Question \d+:") for i in results}


def information_for_questions_to_csv(csv_path, information_for_questions):
    df = pd.read_csv(csv_path, encoding="utf-8")
    start_column = next((i for i, column in enumerate(df.columns) if df[column].isna().all()), 0)
    df = pd.concat([df, pd.DataFrame([
        [None] * start_column + [None if information_for_question[key] == "None" else information_for_question[key]
         for key in range(1, len(information_for_question) + 1)]
        for key, information_for_question in sorted(information_for_questions.items())], columns=df.columns)])
    df.to_csv(csv_path, index=False, encoding="utf-8")


def scripts_for_audio_interpretation(csv_path, notes):
    system_message = get_prompt("script_for_audio_interpretation")
    df = pd.read_csv(csv_path, encoding="utf-8")
    default_notes = """Focus on pivotal findings and their business relevance. Provide explanations for technical terms if any. Decode complex ideas using everyday examples and comparisons. Conclude with key takeaways and a final statement."""
    user_messages = [f"<question>\n{column}\n</question>\n<information_for_question>\n{information_for_question}\n</information_for_question>\n<notes>\n{notes if notes else default_notes}\n</notes>" for column in df.columns if len(information_for_question := "\n\n".join(df[column].dropna().astype(str))) >= 300]
    results = text_chats(["gpt4o_mini_rainbow", "gpt4o_mini_openrouter", "gpt4o_mini_excellence"], system_message, user_messages)
    return {i: f"{df.columns[i-1]}\n{filter_words(extract_text_with_xml_tag(results[i], 'script'), ['*', '首先，', '其次，', '再次，', '最后，', '然而，', '然而', '此外，', '此外', '除此之外，', '总之，', '总而言之，', '总的来说，', '综上所述，', '中共', '中国共产党'])}" for i in results}


def generate_audio_interpretation(txt_path, user_requirements, voice_gender, to_email=None):
    csv_path = f"temp-data/{now_and_choices()}.csv"
    doc_contents = read_txt(txt_path)
    for attempt in range(3):
        try:
            outline, notes = outline_for_audio_interpretation(csv_path, user_requirements, "\n\n".join(doc_contents[key] for key in sorted(doc_contents)))
            if outline and csv_path:
                for attempt in range(3):
                    try:
                        information_for_questions = information_for_audio_interpretation(doc_contents, outline)
                        if information_for_questions:
                            information_for_questions_to_csv(csv_path, information_for_questions)
                            for attempt in range(3):
                                try:
                                    scripts = scripts_for_audio_interpretation(csv_path, notes)
                                    if scripts:
                                        for attempt in range(3):
                                            try:
                                                audios = generate_audios_serially(scripts, voice_gender)
                                                if audios and sum(1 for audio in audios.values() if not audio) / len(audios) < 0.2:
                                                    for attempt in range(3):
                                                        try:
                                                            wav_path = merge_audios(audios)
                                                            if wav_path:
                                                                file_url = upload_and_send(wav_path, to_email)
                                                                return file_url
                                                        except Exception:
                                                            print(f"Step 5 attempt {attempt + 1} failed")
                                                    return None
                                            except Exception:
                                                print(f"Step 4 attempt {attempt + 1} failed")
                                        return None
                                except Exception:
                                    print(f"Step 3 attempt {attempt + 1} failed")
                            return None
                    except Exception:
                        print(f"Step 2 attempt {attempt + 1} failed")
                return None
        except Exception:
            print(f"Step 1 attempt {attempt + 1} failed")
    return None


def calculator(expression):
    getcontext().prec = 50
    try:
        return eval(expression, {"__builtins__": None, "math": math, "Decimal": Decimal})
    except Exception:
        return None
