import datasets
import re

def load_strategyqa():
    """Loads StrategyQA dataset."""
    ds = datasets.load_dataset('tasksource/strategy-qa')
    return ds['train']

def load_bbh_logical_deduction():
    """Loads BBH Logical Deduction (3 objects)."""
    ds = datasets.load_dataset('lukaemon/bbh', 'logical_deduction_three_objects')
    return ds['test']

# Two-shot exemplars: Qwen2.5-Instruct in raw completion mode rambles in numbered
# lists forever and never converges to "So the answer is X" without exemplars.
# These are short, balanced (one Yes, one No), and end with the exact conclusion
# string the regex in parse_strategyqa_answer looks for.
_STRATEGYQA_FEWSHOT = (
    "Q: Could a person with cancer go skydiving?\n"
    "A: Let me work through this step by step.\n"
    "1. Skydiving requires the participant to be in reasonable physical health to jump from an aircraft.\n"
    "2. Cancer varies widely in severity; many patients in remission or with mild forms remain physically active.\n"
    "3. Skydiving operators have general health requirements but do not categorically exclude all cancer patients.\n"
    "So the answer is Yes.\n\n"
    "Q: Are most teenagers fluent in Latin?\n"
    "A: Let me work through this step by step.\n"
    "1. Latin is a classical language that is no longer commonly spoken in daily life.\n"
    "2. Most modern school systems do not require Latin as a core subject, and few teach it at all.\n"
    "3. Even where Latin is offered, only a small fraction of students take it, and fewer still reach fluency.\n"
    "So the answer is No.\n\n"
)


def format_strategyqa_prompt(question: str) -> str:
    """Formats prompt for StrategyQA with two-shot exemplars so the model
    reliably ends with 'So the answer is Yes.' / 'So the answer is No.'"""
    return _STRATEGYQA_FEWSHOT + f"Q: {question}\nA: Let me work through this step by step.\n"

def format_bbh_prompt(input_text: str) -> str:
    """Formats prompt for BBH Logical Deduction."""
    return f"{input_text}\nLet me work through the constraints one by one.\n"

# Single regex used for BOTH parsing the answer and locating the trigger token
# during activation extraction. Order of alternatives matters: more specific
# patterns first so e.g. "Final answer: Yes" doesn't get split as "answer: yes".
ANSWER_TRIGGER_RE = re.compile(
    r"(?i)(?:"
    r"so\s*,?\s*the\s+answer\s+is\s+"        # "so the answer is", "So, the answer is"
    r"|final\s+answer\s*:?\s*"               # "Final answer:" / "Final answer "
    r"|the\s+answer\s+is\s+"                 # bare "the answer is" (no "so")
    r"|answer\s*:\s*"                        # "Answer: "
    r")(yes|no)\b"
)


def parse_strategyqa_answer(text: str) -> int:
    """Parses binary yes/no from output text. Returns 1 for Yes, 0 for No, -1 for invalid.
    Accepts the canonical "So the answer is X" plus three observed Qwen variants:
    "So, the answer is X", bare "the answer is X", and "Final answer: X"."""
    matches = ANSWER_TRIGGER_RE.findall(text)
    if matches:
        return 1 if matches[-1].lower() == 'yes' else 0
    return -1
