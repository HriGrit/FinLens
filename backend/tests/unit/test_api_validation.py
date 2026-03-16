from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.main import ChatRequest


def test_chat_request_rejects_nonpositive_retrieval_top_k() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(query="test", retrieval_top_k=0)


def test_chat_request_rejects_nonpositive_rerank_top_k() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(query="test", rerank_top_k=-1)
