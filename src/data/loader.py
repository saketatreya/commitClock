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

def format_strategyqa_prompt(question: str) -> str:
    """Formats prompt for StrategyQA."""
    return f"Q: {question}\nA: Let me work through this step by step.\n"

def format_bbh_prompt(input_text: str) -> str:
    """Formats prompt for BBH Logical Deduction."""
    return f"{input_text}\nLet me work through the constraints one by one.\n"

def parse_strategyqa_answer(text: str) -> int:
    """Parses binary yes/no from output text. Returns 1 for Yes, 0 for No, -1 for invalid."""
    # Look for "So the answer is Yes." or "So the answer is No."
    # We can be somewhat flexible but need to be careful.
    matches = re.findall(r"(?i)so the answer is (yes|no)", text)
    if matches:
        ans = matches[-1].lower()
        return 1 if ans == 'yes' else 0
    return -1
