# DEEP-FACE-RECOGNITION-USING_-Deep-Learning-CNN-ResNet-SVM-OpenCV-

Designed and implemented a deep face recognition system leveraging CNN-based feature extraction (ResNet) and  SVM classification. Enabled identity verification through embedding similarity and ROC-based threshold selection.  Achieved high accuracy and robustness across varying lighting and pose conditions.


This repository provides an end-to-end Deep Face Recognition system for identity verification:
- Face detection & alignment (MTCNN)
- Face embedding extraction (Inception Resnet v1 pretrained — FaceNet-style)
- Enrollment (build embeddings database)
- Classifier-based identification (SVM) and threshold-based verification (cosine similarity)
- Evaluation scripts (identification & verification metrics)
- A lightweight Flask inference API + simple web form for image upload

Design goals
- Reproducible and practical pipeline for research & prototyping.
- Use high-quality pretrained backbone (facenet-pytorch) to avoid training from scratch.
- Keep components modular so you can replace detectors, embedder, or the classifier.
- Provide instructions and scripts for dataset preparation, training, evaluation, and deployment.

Contents
- README.md (this file)
- requirements.txt — Python dependencies
- utils.py — detection, alignment, embedding utilities
- train_embeddings.py — create embeddings dataset from image folders
- train_classifier.py — train identification classifier (SVM) on embeddings
- evaluate.py — identification and verification evaluation utilities (ROC, EER, TAR@FAR)
- inference_app.py — Flask app serving the model for identification/verification
- demo_web_template.html — minimal web UI used by Flask
- dataset_format.md — guidance on dataset folder structure and public datasets
- models/ — saved artifacts (created at runtime): embeddings.npz, classifier.joblib, label_encoder.joblib

Quick summary of pipeline
1. Prepare dataset: folder per identity containing sample face images.
   data/train/<person_name>/*.jpg
   data/val/<person_name>/*.jpg (optional for evaluation)
2. Run embedding extraction:
   python train_embeddings.py --data_dir data/train --out_path models/embeddings_train.npz
3. Train classifier:
   python train_classifier.py --embeddings models/embeddings_train.npz --out_dir models
   This produces classifier.joblib and label_encoder.joblib.
4. Start inference server:
   python inference_app.py --model_dir models --host 0.0.0.0 --port 5000
   Open http://127.0.0.1:5000 to upload an image and get predicted identity + confidence.
5. Verification:
   Use evaluate.py to compute ROC and choose thresholds, or call the /verify endpoint of the Flask app.

Notes about performance & accuracy
- With a good dataset (≥10 images/person, varied poses/lighting) and using pretrained InceptionResnetV1 embeddings, a simple SVM classifier or nearest-neighbor cosine matching yields strong performance for identification and verification.
- For highly-secure production usage:
  - Use larger and more diverse enrollment images, multi-sample templates per identity.
  - Use liveness detection and anti-spoofing (not covered here).
  - Use an approval workflow (human-in-loop) for enrollment and critical operations.

Ethics, privacy & legal
- Face recognition can have significant privacy and fairness implications. Obtain informed consent from users, follow local laws and regulations, and test for demographic bias.
- This code is for research/prototyping. Do not deploy for high-risk identity decisions without rigorous testing and compliance.

What I included and why
- I used facenet-pytorch (MTCNN + InceptionResnetV1) so you can get reliable embeddings out-of-the-box without training deep backbones yourself.
- The pipeline separates embedding extraction from classifier training so you can re-run classifier experiments quickly, or swap classifiers.
- The Flask app demonstrates how to integrate detection + embedder + classifier for quick demos or lightweight deployments.

What's next (suggested)
- Add liveness/anti-spoofing module (e.g., eye-blink, challenge-response, or a CNN trained for spoof detection).
- Add enrollment UI for adding new identities with multi-image templates.
- Move model serving to a proper service (FastAPI + gunicorn) and add caching for embeddings.
- Add logging, drift monitoring, and scheduled re-enrollment for users with large appearance changes.
