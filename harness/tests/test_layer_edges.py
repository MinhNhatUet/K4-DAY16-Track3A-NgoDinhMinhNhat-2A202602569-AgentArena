"""Regression cases for irregular model output; no public answer fixtures."""

from types import SimpleNamespace

import pytest

from arena.corpus import Corpus, Doc
from harness.layers.citation_checker import CitationChecker
from harness.layers.critic import Critic


def context(body):
    doc = Doc(doc_id="source-a", title="Evidence", body=body, tags=())
    return SimpleNamespace(corpus=Corpus([doc]), observed_text=body)


@pytest.mark.parametrize("doc_id", [[], {}, None, 42])
def test_citation_checker_repairs_invalid_id_type_without_changing_text(doc_id):
    text = "Evidence is quoted exactly, including punctuation."
    ctx = context(text)
    report = {"claims": [{"text": text, "doc_id": doc_id}]}
    result = CitationChecker().after_agent(ctx, report)
    assert result["claims"] == [{"text": text, "doc_id": "source-a"}]
    assert result["citations"] == ["source-a"]


def test_critic_rejects_quote_spanning_document_lines():
    text = "First independent evidence line.\nSecond independent evidence line."
    ctx = context(text)
    report = {"claims": [{"text": text, "doc_id": "source-a"}], "abstain": False}
    result = Critic().after_agent(ctx, report)
    assert result["claims"] == []
    assert result["citations"] == []
    assert result["abstain"] is True


def test_critic_keeps_verbatim_substring_without_normalising_it():
    text = "Evidence  keeps its spacing"
    ctx = context("Prefix: " + text + ".")
    claim = {"text": text, "doc_id": "source-a"}
    report = {"claims": [claim], "abstain": False}
    result = Critic().after_agent(ctx, report)
    assert result["claims"] == [claim]
    assert result["claims"][0]["text"] == text
    assert result["abstain"] is False


def test_critic_rejects_tool_wrapper_text_not_supported_by_document():
    ctx = context("Actual document evidence.")
    wrapper = "Tool metadata is not a document quotation."
    ctx.observed_text += "\n" + wrapper
    report = {"claims": [{"text": wrapper, "doc_id": "source-a"}]}
    result = Critic().after_agent(ctx, report)
    assert result["claims"] == []
    assert result["abstain"] is True


def test_critic_leaves_misattribution_to_citation_checker():
    text = "Evidence appears on one line in the document."
    ctx = context(text)
    claim = {"text": text, "doc_id": "wrong-source"}
    result = Critic().after_agent(ctx, {"claims": [claim]})
    assert result["claims"] == [claim]


def test_citation_checker_does_not_repair_from_unobserved_document():
    text = "Unobserved evidence must not become a citation."
    ctx = context(text)
    ctx.observed_text = "A different observation."
    report = {"claims": [{"text": text, "doc_id": []}]}
    result = CitationChecker().after_agent(ctx, report)
    assert result["claims"] == [{"text": text, "doc_id": []}]
    assert result["citations"] == []


def test_injection_guard_scrubs_canary_from_every_field_and_drops_its_claims():
    from arena.corpus import INJECTION_CANARY
    from harness.layers.injection_guard import InjectionGuard

    report = {
        "answer": f"{INJECTION_CANARY} ok",
        "verdict": f"chọn (b) {INJECTION_CANARY}",
        "claims": [{"text": f"x {INJECTION_CANARY}", "doc_id": "d"}, {"text": "y", "doc_id": "d"}],
    }
    result = InjectionGuard().after_agent(None, report)
    assert INJECTION_CANARY not in str(result)
    assert result["claims"] == [{"text": "y", "doc_id": "d"}]


def test_critic_adds_the_addendum_for_a_real_model_only():
    from arena.model import MockModel
    from harness.agent import REAL_MODEL_PROMPT_ADDENDUM

    messages = [{"role": "system", "content": "base"}, {"role": "user", "content": "q"}]
    real = Critic().before_model(SimpleNamespace(model=object()), messages)
    assert REAL_MODEL_PROMPT_ADDENDUM.strip() in real[0]["content"]
    assert messages[0]["content"] == "base"  # không sửa list gốc
    mock = MockModel(Corpus.generate(seed=42), seed=1)
    assert Critic().before_model(SimpleNamespace(model=mock), messages) is messages
