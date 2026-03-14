"""
Guardrails — input + output validation layer.

Flow:
  raw input → InputGuardrail → agent → OutputGuardrail → response

Checks:
  Input:  prompt injection, PII detection, toxic content, message length
  Output: PII leakage, hallucination signals, empty response, off-topic
"""

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class GuardrailAction(str, Enum):
    PASS    = "pass"
    BLOCK   = "block"
    REDACT  = "redact"
    WARN    = "warn"


@dataclass
class GuardrailResult:
    action:  GuardrailAction
    reason:  Optional[str] = None
    safe_content: Optional[str] = None   # redacted version if action=REDACT
    check_name: Optional[str] = None


# ─────────────────────────────────────────────
# PII patterns
# ─────────────────────────────────────────────

PII_PATTERNS = {
    "credit_card":   r"\b(?:\d[ -]?){13,16}\b",
    "ssn":           r"\b\d{3}-\d{2}-\d{4}\b",
    "email":         r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone":         r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ip_address":    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}

# ─────────────────────────────────────────────
# Prompt injection patterns
# ─────────────────────────────────────────────

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions",
    r"you\s+are\s+now\s+(?:a\s+)?(?:dan|evil|unrestricted)",
    r"forget\s+(?:all\s+)?(?:your\s+)?(?:previous\s+)?instructions",
    r"system\s*:\s*you\s+are",
    r"<\s*system\s*>",
    r"\[INST\]",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"pretend\s+you\s+(have\s+no\s+)?(restrictions|rules|guidelines)",
    r"act\s+as\s+if\s+you\s+(were|are)\s+(not|an?\s+unrestricted)",
]

# ─────────────────────────────────────────────
# Toxic content patterns  
# ─────────────────────────────────────────────

TOXIC_PATTERNS = [
    r"\b(kill|murder|harm|hurt|destroy|attack)\s+(you|yourself|me|them|people)\b",
    r"\b(f+u+c+k|s+h+i+t|b+i+t+c+h|a+s+s+h+o+l+e)\b",
    r"\b(hate|despise)\s+(you|this|everything)\b",
]


# ─────────────────────────────────────────────
# Input Guardrail
# ─────────────────────────────────────────────

class InputGuardrail:
    """
    Validates and sanitizes incoming user messages before
    they reach the LangGraph agent.
    """

    MAX_LENGTH = 5000
    MIN_LENGTH = 2

    def run(self, message: str) -> GuardrailResult:
        """Run all input checks. Returns first failure or PASS."""

        checks = [
            self._check_length,
            self._check_injection,
            self._check_toxic,
            self._check_pii,
        ]

        for check in checks:
            result = check(message)
            if result.action != GuardrailAction.PASS:
                logger.warning(f"Input guardrail triggered: {result.check_name} — {result.reason}")
                return result

        return GuardrailResult(action=GuardrailAction.PASS)

    def _check_length(self, message: str) -> GuardrailResult:
        if len(message) < self.MIN_LENGTH:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                reason="Message too short to process.",
                check_name="length_min"
            )
        if len(message) > self.MAX_LENGTH:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                reason=f"Message exceeds maximum length of {self.MAX_LENGTH} characters.",
                check_name="length_max"
            )
        return GuardrailResult(action=GuardrailAction.PASS)

    def _check_injection(self, message: str) -> GuardrailResult:
        lower = message.lower()
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, lower, re.IGNORECASE):
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    reason="Message contains prompt injection attempt.",
                    check_name="prompt_injection"
                )
        return GuardrailResult(action=GuardrailAction.PASS)

    def _check_toxic(self, message: str) -> GuardrailResult:
        for pattern in TOXIC_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    reason="Message contains inappropriate content.",
                    check_name="toxic_content"
                )
        return GuardrailResult(action=GuardrailAction.PASS)

    def _check_pii(self, message: str) -> GuardrailResult:
        """Redact PII found in input — don't block, just clean."""
        redacted = message
        found_pii = []

        for pii_type, pattern in PII_PATTERNS.items():
            if pii_type == "email":
                continue  # allow emails in support messages
            matches = re.findall(pattern, message)
            if matches:
                found_pii.append(pii_type)
                redacted = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", redacted)

        if found_pii:
            logger.info(f"PII redacted from input: {found_pii}")
            return GuardrailResult(
                action=GuardrailAction.REDACT,
                reason=f"PII detected and redacted: {', '.join(found_pii)}",
                safe_content=redacted,
                check_name="pii_input"
            )

        return GuardrailResult(action=GuardrailAction.PASS)


# ─────────────────────────────────────────────
# Output Guardrail
# ─────────────────────────────────────────────

class OutputGuardrail:
    """
    Validates agent response before sending to user.
    Catches PII leakage, empty responses, hallucination signals.
    """

    HALLUCINATION_SIGNALS = [
        r"as of my (knowledge cutoff|last update|training data)",
        r"I (don't|do not) have (access to|information about) real.time",
        r"I (cannot|can't) browse the internet",
        r"as an AI (language model|assistant)",
        r"I (was|am) trained (on|by)",
    ]

    FALLBACK_RESPONSE = (
        "I'm sorry, I wasn't able to generate a proper response. "
        "Please try rephrasing your question or contact support directly."
    )

    def run(self, response: str, context: dict = None) -> GuardrailResult:
        """Run all output checks."""

        checks = [
            self._check_empty,
            self._check_pii_leakage,
            self._check_hallucination_signals,
        ]

        for check in checks:
            result = check(response)
            if result.action != GuardrailAction.PASS:
                logger.warning(f"Output guardrail triggered: {result.check_name} — {result.reason}")
                return result

        return GuardrailResult(action=GuardrailAction.PASS, safe_content=response)

    def _check_empty(self, response: str) -> GuardrailResult:
        if not response or len(response.strip()) < 10:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                reason="Agent returned empty or too-short response.",
                safe_content=self.FALLBACK_RESPONSE,
                check_name="empty_response"
            )
        return GuardrailResult(action=GuardrailAction.PASS)

    def _check_pii_leakage(self, response: str) -> GuardrailResult:
        """Redact any PII that leaked into the response."""
        redacted = response
        found_pii = []

        for pii_type, pattern in PII_PATTERNS.items():
            matches = re.findall(pattern, response)
            if matches:
                found_pii.append(pii_type)
                redacted = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", redacted)

        if found_pii:
            logger.warning(f"PII leakage detected in output: {found_pii}")
            return GuardrailResult(
                action=GuardrailAction.REDACT,
                reason=f"PII redacted from response: {', '.join(found_pii)}",
                safe_content=redacted,
                check_name="pii_output"
            )

        return GuardrailResult(action=GuardrailAction.PASS)

    def _check_hallucination_signals(self, response: str) -> GuardrailResult:
        """Warn if response contains signals that LLM is hallucinating
        or going off-topic from its support role."""
        for pattern in self.HALLUCINATION_SIGNALS:
            if re.search(pattern, response, re.IGNORECASE):
                # Don't block — just warn and log for eval tracking
                logger.warning(f"Hallucination signal detected in output: {pattern}")
                return GuardrailResult(
                    action=GuardrailAction.WARN,
                    reason="Response may contain off-topic AI behaviour.",
                    safe_content=response,
                    check_name="hallucination_signal"
                )
        return GuardrailResult(action=GuardrailAction.PASS)


# ─────────────────────────────────────────────
# Singletons
# ─────────────────────────────────────────────

input_guardrail  = InputGuardrail()
output_guardrail = OutputGuardrail()