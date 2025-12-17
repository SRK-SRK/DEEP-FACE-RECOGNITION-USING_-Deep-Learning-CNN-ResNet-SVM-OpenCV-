#!/usr/bin/env python3
"""
Create embeddings dataset from a folder of images organized by identity.

Usage:
  python train_embeddings.py --data_dir data/train --out_path models/embeddings_train.npz --verbose

Output:
  models/embeddings_train.npz  (embeddings, labels, paths)
"""
import argparse
import os
from utils import batch_extract_embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Folder with subfolders per identity')
    parser.add_argument('--out_path', type=str, default='models/embeddings_train.npz')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or '.', exist_ok=True)
    batch_extract_embeddings(args.data_dir, args.out_path, verbose=args.verbose)

if __name__ == "__main__":
    main()