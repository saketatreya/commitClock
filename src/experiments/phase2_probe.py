import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from tqdm import tqdm
from scipy.stats import bootstrap

from src.config import PHASE1_OUT_DIR, PHASE2_OUT_DIR, NUM_FRACTIONAL_POSITIONS

def load_phase1_data():
    file_path = PHASE1_OUT_DIR / "strategyqa_activations.pkl"
    if not file_path.exists():
        print(f"Data file not found: {file_path}")
        return []
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def train_probes_for_data(data, label_key):
    """
    Trains logistic regression probes for each (position, layer) cell.
    label_key is 'model_answer' (Probe A) or 'correct_label' (Probe B)
    """
    if not data:
        return None
        
    num_layers = data[0]['activations'].shape[1]
    
    # Organize data into shape [num_samples, num_pos, num_layers, d_model]
    X = np.stack([d['activations'] for d in data]) # [N, 10, L, D]
    y = np.array([d[label_key] for d in data])     # [N]
    
    # We want a heatmap of shape [num_layers, num_pos]
    aurocs = np.zeros((num_layers, NUM_FRACTIONAL_POSITIONS))
    
    unique_classes, class_counts = np.unique(y, return_counts=True)
    if len(unique_classes) > 1:
        n_splits = min(5, min(class_counts))
        if n_splits < 2:
            print(f"Warning: Not enough samples for CV. Skipping CV for {label_key}.")
            return None
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv = None
    
    print(f"Training probes for {label_key}...")
    for pos in tqdm(range(NUM_FRACTIONAL_POSITIONS), desc="Positions"):
        for layer in range(num_layers):
            X_cell = X[:, pos, layer, :] # [N, D]
            
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            
            # To handle cases where one class might be very rare or only one class exists
            if len(np.unique(y)) > 1:
                scores = cross_val_score(clf, X_cell, y, cv=cv, scoring='roc_auc', n_jobs=-1)
                aurocs[layer, pos] = np.mean(scores)
            else:
                aurocs[layer, pos] = 0.5
                
    return aurocs

def plot_commitment_clock(aurocs_a, aurocs_b, title_suffix=""):
    num_layers, num_pos = aurocs_a.shape
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Probe A (Model Answer)
    im1 = ax1.imshow(aurocs_a, origin='lower', aspect='auto', cmap='viridis', vmin=0.4, vmax=1.0)
    ax1.set_title(f"Probe A: Predict Model Answer {title_suffix}")
    ax1.set_xlabel("Fractional Position in Reasoning Chain (%)")
    ax1.set_ylabel("Layer")
    ax1.set_xticks(range(num_pos))
    ax1.set_xticklabels([f"{int(p*100/(num_pos-1))}%" for p in range(num_pos)])
    fig.colorbar(im1, ax=ax1, label="AUROC")
    
    # Probe B (Ground Truth)
    im2 = ax2.imshow(aurocs_b, origin='lower', aspect='auto', cmap='viridis', vmin=0.4, vmax=1.0)
    ax2.set_title(f"Probe B: Predict Ground Truth {title_suffix}")
    ax2.set_xlabel("Fractional Position in Reasoning Chain (%)")
    ax2.set_ylabel("Layer")
    ax2.set_xticks(range(num_pos))
    ax2.set_xticklabels([f"{int(p*100/(num_pos-1))}%" for p in range(num_pos)])
    fig.colorbar(im2, ax=ax2, label="AUROC")
    
    plt.tight_layout()
    plt.savefig(PHASE2_OUT_DIR / f"commitment_clock{title_suffix.replace(' ', '_').lower()}.png")
    plt.close()

def plot_marginal_clock(aurocs_a_easy, aurocs_a_hard):
    num_pos = aurocs_a_easy.shape[1]
    
    # Average across layers
    marg_easy = np.mean(aurocs_a_easy, axis=0)
    marg_hard = np.mean(aurocs_a_hard, axis=0)
    
    positions = [p*100/(num_pos-1) for p in range(num_pos)]
    
    plt.figure(figsize=(8, 6))
    plt.plot(positions, marg_easy, marker='o', label="Easy Questions")
    plt.plot(positions, marg_hard, marker='s', label="Hard Questions")
    plt.axhline(y=0.5, color='r', linestyle='--', label='Chance (0.5)')
    
    plt.title("Marginal Commitment Clock (Layer-Averaged AUROC)")
    plt.xlabel("Fractional Position in Reasoning Chain (%)")
    plt.ylabel("Average AUROC (Probe A)")
    plt.legend()
    plt.grid(True)
    plt.savefig(PHASE2_OUT_DIR / "marginal_clock.png")
    plt.close()

def run_phase2():
    data = load_phase1_data()
    if not data:
        return
        
    print(f"Loaded {len(data)} examples.")
    
    # Stratify by difficulty (approximated here by whether model was correct, or we can use another metric)
    # The paper says: "easy (model gets correct >65% on similar questions, approximated by free-generation accuracy)"
    # We'll approximate 'easy' as cases where the model got it right, 'hard' as where it got it wrong.
    # Note: In a real run, this would be computed per-question over multiple samples.
    
    easy_data = [d for d in data if d['model_answer'] == d['correct_label']]
    hard_data = [d for d in data if d['model_answer'] != d['correct_label']]
    
    print("Training all data...")
    aurocs_a_all = train_probes_for_data(data, 'model_answer')
    aurocs_b_all = train_probes_for_data(data, 'correct_label')
    if aurocs_a_all is not None and aurocs_b_all is not None:
        plot_commitment_clock(aurocs_a_all, aurocs_b_all, title_suffix="")

    print("Training easy data...")
    aurocs_a_easy = train_probes_for_data(easy_data, 'model_answer')
    aurocs_b_easy = train_probes_for_data(easy_data, 'correct_label')
    if aurocs_a_easy is not None and aurocs_b_easy is not None:
        plot_commitment_clock(aurocs_a_easy, aurocs_b_easy, title_suffix=" - Easy")

    print("Training hard data...")
    aurocs_a_hard = train_probes_for_data(hard_data, 'model_answer')
    aurocs_b_hard = train_probes_for_data(hard_data, 'correct_label')
    if aurocs_a_hard is not None and aurocs_b_hard is not None:
        plot_commitment_clock(aurocs_a_hard, aurocs_b_hard, title_suffix=" - Hard")
        
    if aurocs_a_easy is not None and aurocs_a_hard is not None:
        plot_marginal_clock(aurocs_a_easy, aurocs_a_hard)
        
    print("Phase 2 complete.")

if __name__ == "__main__":
    run_phase2()
