"""
Smart Escalation: know when to hand off to a human agent.

An agent that sells must also know when to STOP and hand the chat to a person.
Handoff fires when:
  1. Customer explicitly asks for a human (handoff keywords).
  2. Customer is angry (emotion detection).
  3. Input guardrail flags prompt injection or abuse.
  4. Agent confidence drops below a threshold.

Reference: Ecbot (github.com/blackmouse572/ecbot)
"""
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class HandoffReason(str, Enum):
    CUSTOMER_REQUEST = "customer_requested_human"
    ANGRY_CUSTOMER = "angry_customer"
    PROMPT_INJECTION = "prompt_injection"
    LOW_CONFIDENCE = "low_confidence"
    ABUSE = "abuse"


@dataclass
class EscalationDecision:
    """Result of an escalation check."""
    should_handoff: bool = False
    reasons: list[HandoffReason] = field(default_factory=list)
    summary: str = ""

    def add(self, reason: HandoffReason, detail: str = ""):
        self.should_handoff = True
        if reason not in self.reasons:
            self.reasons.append(reason)
        if detail:
            self.summary = (self.summary + "; " + detail).strip("; ")


class SmartEscalation:
    """Evaluates whether a turn should be escalated to a human."""

    # Phrases that explicitly request a human
    HANDOFF_KEYWORDS = [
        r"speak to (a |an |the )?(human|agent|person|representative)",
        r"talk to (a |an |the )?(human|person|representative)",
        r"real person",
        r"human agent",
        r"transfer me",
        r"give me a human",
        r"转人工",
        r"人工客服",
        r"找真人",
    ]

    # Anger signals
    ANGER_KEYWORDS = [
        "angry", "furious", "outraged", "terrible", "worst",
        "ridiculous", "unacceptable", "disgusting", "horrible",
        "complaint", "lawyer", "sue", "refund immediately",
        "气死", "愤怒", "投诉", "太差", "垃圾", "骗子",
    ]

    # Prompt injection / abuse signals
    INJECTION_PATTERNS = [
        r"ignore (all |previous |prior )?(instructions?|prompts?)",
        r"disregard (all |previous )?(instructions?|prompts?)",
        r"you are now",
        r"system prompt",
        r"reveal your (instructions|prompt|system)",
        r"忽略(之前|以上|所有).{0,4}(指令|提示)",
        r"你现在是",
    ]

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        anger_threshold: int = 2,
        enable_injection_detection: bool = True,
    ):
        self.confidence_threshold = confidence_threshold
        self.anger_threshold = anger_threshold
        self.enable_injection_detection = enable_injection_detection
        self._handoff_regex = re.compile(
            "|".join(self.HANDOFF_KEYWORDS), re.IGNORECASE
        )
        self._injection_regex = re.compile(
            "|".join(self.INJECTION_PATTERNS), re.IGNORECASE
        )

    def check(
        self,
        user_message: str,
        agent_confidence: Optional[float] = None,
        agent_tags: Optional[list[str]] = None,
        conversation_history: Optional[list[str]] = None,
    ) -> EscalationDecision:
        """
        Evaluate one turn for escalation.

        user_message: latest customer message
        agent_confidence: model self-reported confidence (0-1), if available
        agent_tags: tags applied to the customer (e.g. ['Angry'])
        conversation_history: recent messages, for cumulative anger detection
        """
        decision = EscalationDecision()
        text = user_message or ""
        lower = text.lower()

        # 1. Explicit human request
        if self._handoff_regex.search(text):
            decision.add(
                HandoffReason.CUSTOMER_REQUEST,
                "customer explicitly requested a human",
            )

        # 2. Anger detection
        anger_count = sum(1 for kw in self.ANGER_KEYWORDS if kw in lower)
        # Check tags
        if agent_tags and any(t.lower() in ("angry", "furious") for t in agent_tags):
            anger_count += 1
        # Cumulative anger from history
        if conversation_history:
            anger_count += sum(
                1 for m in conversation_history
                if any(kw in m.lower() for kw in self.ANGER_KEYWORDS)
            )

        if anger_count >= self.anger_threshold:
            decision.add(
                HandoffReason.ANGRY_CUSTOMER,
                f"anger signals: {anger_count}",
            )

        # 3. Prompt injection / abuse
        if self.enable_injection_detection:
            if self._injection_regex.search(text):
                decision.add(
                    HandoffReason.PROMPT_INJECTION,
                    "prompt injection pattern detected",
                )

        # 4. Low confidence
        if agent_confidence is not None and agent_confidence < self.confidence_threshold:
            decision.add(
                HandoffReason.LOW_CONFIDENCE,
                f"confidence {agent_confidence:.2f} < {self.confidence_threshold}",
            )

        return decision

    def build_handoff_summary(self, conversation_id: str, decision: EscalationDecision) -> str:
        """Build the summary shown to the human operator taking over."""
        reasons = ", ".join(r.value for r in decision.reasons)
        return (
            f"[Handoff {conversation_id}] reasons: {reasons}. "
            f"details: {decision.summary}"
        )
