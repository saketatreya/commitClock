import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from tqdm import tqdm

from src.config import PHASE1_OUT_DIR, PHASE6_OUT_DIR, NUM_FRACTIONAL_POSITIONS

def load_phase1_data():
    file_path = PHASE1_OUT_DIR / "strategyqa_activations.pkl"
    if not file_path.exists():
        print(f"Data file not found: {file_path}")
        return []
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def train_probes_for_nonlinearity(data):
    if not data:
        return None, None
        
    num_layers = data[0]['activations'].shape[1]
    
    X = np.stack([d['activations'] for d in data]) # [N, 10, L, D]
    y = np.array([d['model_answer'] for d in data]) # [N]
    
    lr_aurocs = np.zeros((num_layers, NUM_FRACTIONAL_POSITIONS))
    mlp_aurocs = np.zeros((num_layers, NUM_FRACTIONAL_POSITIONS))
    
    unique_classes, class_counts = np.unique(y, return_counts=True)
    if len(unique_classes) > 1:
        n_splits = min(5, min(class_counts))
        if n_splits < 2:
            print(f"Warning: Not enough samples for CV. Skipping CV.")
            return None, None
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv = None
    
    print("Training LR and MLP probes for nonlinearity index...")
    for pos in tqdm(range(NUM_FRACTIONAL_POSITIONS), desc="Positions"):
        for layer in range(num_layers):
            X_cell = X[:, pos, layer, :] # [N, D]
            
            lr_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            mlp_clf = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', max_iter=1000, random_state=42)
            
            if len(np.unique(y)) > 1:
                lr_scores = cross_val_score(lr_clf, X_cell, y, cv=cv, scoring='roc_auc', n_jobs=-1)
                mlp_scores = cross_val_score(mlp_clf, X_cell, y, cv=cv, scoring='roc_auc', n_jobs=-1)
                
                lr_aurocs[layer, pos] = np.mean(lr_scores)
                mlp_aurocs[layer, pos] = np.mean(mlp_scores)
            else:
                lr_aurocs[layer, pos] = 0.5
                mlp_aurocs[layer, pos] = 0.5
                
    return lr_aurocs, mlp_aurocs

def plot_nonlinearity_index(lr_aurocs, mlp_aurocs):
    num_layers, num_pos = lr_aurocs.shape
    
    nonlinearity_index = mlp_aurocs - lr_aurocs
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(nonlinearity_index, origin='lower', aspect='auto', cmap='coolwarm')
    
    plt.title("Nonlinearity Index (MLP AUROC - LR AUROC)")
    plt.xlabel("Fractional Position in Reasoning Chain (%)")
    plt.ylabel("Layer")
    plt.xticks(range(num_pos), [f"{int(p*100/(num_pos-1))}%" for p in range(num_pos)])
    plt.colorbar(im, label="AUROC Difference")
    
    plt.tight_layout()
    plt.savefig(PHASE6_OUT_DIR / "nonlinearity_index.png")
    plt.close()

def run_phase6():
    data = load_phase1_data()
    if not data:
        return
        
    lr_aurocs, mlp_aurocs = train_probes_for_nonlinearity(data)
    if lr_aurocs is not None and mlp_aurocs is not None:
        plot_nonlinearity_index(lr_aurocs, mlp_aurocs)
        
        # Save the matrices
        with open(PHASE6_OUT_DIR / "nonlinearity_results.pkl", "wb") as f:
            pickle.dump({"lr_aurocs": lr_aurocs, "mlp_aurocs": mlp_aurocs}, f)
            
    print("Phase 6 complete.")

if __name__ == "__main__":
    run_phase6()
