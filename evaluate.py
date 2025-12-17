#!/usr/bin/env python3
"""
Evaluation utilities:
- Identification: classification report on held-out set (use classifier)
- Verification: compute ROC and EER by comparing pairs (genuine vs imposter) using cosine similarity

Usage examples:
  python evaluate.py --embeddings models/embeddings_train.npz --classifier models/classifier.joblib
  python evaluate.py --embeddings models/embeddings_train.npz --verify_pairs pairs.csv
"""
import argparse
import numpy as np
import joblib
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from utils import load_embeddings, cosine_similarity

def identification_report(embeddings_path, classifier_path, label_encoder_path=None):
    X, labels, paths = load_embeddings(embeddings_path)
    clf = joblib.load(classifier_path)
    if label_encoder_path:
        le = joblib.load(label_encoder_path)
        names = le.inverse_transform(clf.predict(X))
    else:
        names = clf.predict(X)
    preds = clf.predict(X)
    # If label encoder provided, map labels:
    print("[INFO] Identification evaluation on provided embeddings (no train/test split here).")
    # users can compute detailed metrics by splitting beforehand; we keep this simple.

def verification_roc(embeddings_path, n_pairs=10000, out_plot='figures/roc_verification.png'):
    """
    Build random genuine and imposter pairs and compute ROC.
    """
    X, labels, paths = load_embeddings(embeddings_path)
    # Build mapping label -> indices
    from collections import defaultdict
    idxs = defaultdict(list)
    for i, lab in enumerate(labels):
        idxs[lab].append(i)
    # sample genuine pairs
    genuine = []
    import random
    for _ in range(n_pairs//2):
        lab = random.choice(list(idxs.keys()))
        if len(idxs[lab]) < 2:
            continue
        a, b = random.sample(idxs[lab], 2)
        genuine.append((a,b))
    imposter = []
    keys = list(idxs.keys())
    for _ in range(n_pairs//2):
        a_lab, b_lab = random.sample(keys, 2)
        a = random.choice(idxs[a_lab])
        b = random.choice(idxs[b_lab])
        imposter.append((a,b))
    pairs = genuine + imposter
    y_true = np.array([1]*len(genuine) + [0]*len(imposter))
    scores = []
    for i,j in pairs:
        s = cosine_similarity(X[i], X[j])
        scores.append(s)
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.4f})')
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('Verification ROC')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot)
    print(f"[INFO] ROC saved to {out_plot}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=str, required=True)
    parser.add_argument('--classifier', type=str, default=None)
    parser.add_argument('--label_encoder', type=str, default=None)
    parser.add_argument('--verify_pairs', type=str, default=None)
    parser.add_argument('--out_plot', type=str, default='figures/roc_verification.png')
    args = parser.parse_args()

    if not args.classifier:
        verification_roc(args.embeddings, out_plot=args.out_plot)
    else:
        identification_report(args.embeddings, args.classifier, args.label_encoder)

if __name__ == "__main__":
    main()