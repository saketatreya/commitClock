import pickle
import random
from tqdm import tqdm

from src.config import PHASE1_OUT_DIR, PHASE3_OUT_DIR

def load_phase1_data():
    file_path = PHASE1_OUT_DIR / "strategyqa_activations.pkl"
    if not file_path.exists():
        print(f"Data file not found: {file_path}")
        return []
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def truncate_and_append(wrong_text: str, correct_label: int) -> str:
    """
    Truncates a wrong reasoning chain randomly and appends a correct conclusion.
    """
    # Assuming text looks like "... Let me work through this step by step.\n[reasoning]\nSo the answer is No."
    ans_str = "Yes" if correct_label == 1 else "No"
    trigger_str = "so the answer is"
    
    idx = wrong_text.lower().rfind(trigger_str)
    if idx == -1:
        # Fallback if parsing failed
        return wrong_text + f"\nSo the answer is {ans_str}."
        
    reasoning_part = wrong_text[:idx]
    
    # Split reasoning into sentences and keep only the first half
    sentences = reasoning_part.split('.')
    keep_len = max(1, len(sentences) // 2)
    truncated = '.'.join(sentences[:keep_len]) + '.'
    
    return truncated + f"\nSo the answer is {ans_str}."

def run_phase3():
    data = load_phase1_data()
    if not data:
        return
        
    print(f"Loaded {len(data)} examples from Phase 1.")
    
    # We need to construct 4 conditions for each question that has a Condition 1 (correct reasoning)
    correct_samples = [d for d in data if d['model_answer'] == d['correct_label']]
    wrong_samples = [d for d in data if d['model_answer'] != d['correct_label']]
    
    if not wrong_samples:
        print("Warning: No wrong samples found in data. Condition 3 will be empty or faked.")
    
    results = []
    
    for item in tqdm(correct_samples, desc="Constructing Forced Branches"):
        q_id = item['question_id']
        prompt = item['prompt']
        correct_label = item['correct_label']
        ans_str = "Yes" if correct_label == 1 else "No"
        
        # Condition 1: Natural correct reasoning
        cond1_text = item['generated_text']
        
        # Condition 3: Wrong reasoning reaching wrong answer
        # Ideally, we find a wrong reasoning for the SAME question.
        # But if not available, we use a wrong reasoning from another question.
        same_q_wrong = [w for w in wrong_samples if w['question_id'] == q_id]
        if same_q_wrong:
            cond3_text = same_q_wrong[0]['generated_text']
        elif wrong_samples:
            # Pick a random wrong sample
            cond3_text = random.choice(wrong_samples)['generated_text']
            # We must replace its conclusion with the wrong answer for THIS question.
            # But the easiest way is to just use its reasoning part and append the wrong answer.
            idx = cond3_text.lower().rfind("so the answer is")
            if idx != -1:
                wrong_ans_str = "No" if correct_label == 1 else "Yes"
                cond3_text = cond3_text[:idx] + f"So the answer is {wrong_ans_str}."
        else:
            cond3_text = cond1_text.replace("not", "").replace("is", "is not") # Hacky fallback
            
        # Condition 2: Wrong reasoning reaching correct answer
        cond2_text = truncate_and_append(cond3_text, correct_label)
        
        # Condition 4: No reasoning (direct answer)
        # Assuming prompt ends with "Let me work through this step by step.\n"
        # We strip that and just ask for answer.
        # Actually, Phase 1 formatted it. We can just say:
        cond4_text = prompt.replace("Let me work through this step by step.\n", "") + f"So the answer is {ans_str}."
        
        record = {
            "question_id": q_id,
            "correct_label": correct_label,
            "cond1": cond1_text,
            "cond2": cond2_text,
            "cond3": cond3_text,
            "cond4": cond4_text,
        }
        results.append(record)
        
    with open(PHASE3_OUT_DIR / "forced_branches.pkl", "wb") as f:
        pickle.dump(results, f)
        
    print(f"Phase 3 complete. Constructed forced branches for {len(results)} questions.")

if __name__ == "__main__":
    run_phase3()
