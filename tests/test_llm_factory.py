"""Dependency-free unit tests; no network, model downloads, or API calls."""
import os
import sys
import types
import unittest
from unittest.mock import patch

from llm_providers.llm_factory import get_llm


class FakeChat:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def fake_module(name, class_name):
    module = types.ModuleType(name)
    setattr(module, class_name, FakeChat)
    return module


class ModelFactoryTests(unittest.TestCase):
    def test_default_is_local(self):
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {
            "langchain_ollama": fake_module("langchain_ollama", "ChatOllama")
        }):
            self.assertEqual(get_llm().kwargs["model"], "qwen3.5:4b")

    def test_premium_denied_without_permission(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(PermissionError):
                get_llm("premium")

    def test_openai_selects_paid_only_when_authorized(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder"}, clear=True), patch.dict(sys.modules, {
            "langchain_openai": fake_module("langchain_openai", "ChatOpenAI")
        }):
            self.assertEqual(
                get_llm("premium", "analyst", allow_external=True).kwargs["model"],
                "gpt-5.5",
            )

    def test_switch_to_claude_without_code_change(self):
        with patch.dict(os.environ, {
            "SM_LLM_PREMIUM_PROVIDER": "anthropic",
            "SM_LLM_PREMIUM_MODEL": "claude-sonnet-5-5",
            "ANTHROPIC_API_KEY": "test-placeholder",
        }, clear=True), patch.dict(sys.modules, {
            "langchain_anthropic": fake_module("langchain_anthropic", "ChatAnthropic")
        }):
            self.assertEqual(
                get_llm("premium", "analyst", allow_external=True).kwargs["model"],
                "claude-sonnet-5-5",
            )

    def test_no_paid_credentials_fails_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                get_llm("premium", "analyst", allow_external=True)

    def test_invalid_profile_fails(self):
        with self.assertRaises(ValueError):
            get_llm("unlisted")


if __name__ == "__main__":
    unittest.main()
