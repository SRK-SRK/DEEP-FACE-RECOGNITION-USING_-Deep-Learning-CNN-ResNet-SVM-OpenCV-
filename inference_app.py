#!/usr/bin/env python3
"""
Flask app for identification and verification.

Endpoints:
- GET / -> basic upload form
- POST /predict -> form upload (file) -> returns predicted identity + confidence
- POST /api/predict -> JSON or multipart: {'image': file} returns JSON {label, confidence, distances...}
- POST /api/verify -> accepts two images and returns similarity score and verdict (match if score >= threshold)

Notes:
- Requires models/classifier.joblib, models/label_encoder.joblib and the facenet artifacts (downloaded on first run by facenet-pytorch)

"""

import os
import io
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
from PIL import Image
from utils import get_mtcnn, get_resnet, extract_aligned_face, get_embedding, image_to_embedding, cosine_similarity

app = Flask(__name__)

# Paths to saved artifacts
MODEL_DIR = os.environ.get('MODEL_DIR', 'models')
CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'classifier.joblib')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.joblib')

# Lazy-load
clf = None
label_encoder = None
mtcnn = None
resnet = None

UNKNOWN_THRESHOLD = 0.0  # confidence threshold for "unknown" vs known label (for classifier probability)
COSINE_THRESHOLD = 0.45  # for verification; cosine similarity threshold (adjust via ROC)

def load_artifacts():
    global clf, label_encoder, mtcnn, resnet
    if clf is None and os.path.exists(CLASSIFIER_PATH):
        clf = joblib.load(CLASSIFIER_PATH)
    if label_encoder is None and os.path.exists(LABEL_ENCODER_PATH):
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    if mtcnn is None:
        mtcnn = get_mtcnn()
    if resnet is None:
        resnet = get_resnet()

@app.route('/')
def index():
    return render_template('demo_web_template.html')

def prepare_image_from_request(req):
    # support file upload (multipart/form-data) with key 'image'
    if 'image' in req.files:
        f = req.files['image']
        img = Image.open(f.stream).convert('RGB')
        return img
    # support JSON with base64 image (optional)
    data = req.get_json(silent=True)
    if data and 'image_b64' in data:
        import base64
        decoded = base64.b64decode(data['image_b64'])
        img = Image.open(io.BytesIO(decoded)).convert('RGB')
        return img
    return None

@app.route('/predict', methods=['POST'])
def predict_view():
    load_artifacts()
    img = prepare_image_from_request(request)
    if img is None:
        return "No image provided", 400
    face = extract_aligned_face(img, mtcnn=mtcnn)
    if face is None:
        return "No face detected", 400
    emb = get_embedding(face, resnet=resnet)
    # classifier expects shape (N,512)
    if clf is None:
        return "Model not found. Train classifier first.", 500
    proba = clf.predict_proba([emb])[0]
    pred_idx = np.argmax(proba)
    confidence = float(proba[pred_idx])
    # map index -> label name if label encoder exists
    label = None
    if label_encoder is not None:
        label = label_encoder.inverse_transform([pred_idx])[0]
    else:
        label = str(pred_idx)
    # apply unknown threshold
    if confidence < UNKNOWN_THRESHOLD:
        label = "unknown"
    # return simple HTML
    return f"Predicted: {label} (confidence={confidence:.3f})"

@app.route('/api/predict', methods=['POST'])
def api_predict():
    load_artifacts()
    img = prepare_image_from_request(request)
    if img is None:
        return jsonify({'error':'No image provided'}), 400
    face = extract_aligned_face(img, mtcnn=mtcnn)
    if face is None:
        return jsonify({'error':'No face detected'}), 400
    emb = get_embedding(face, resnet=resnet)
    if clf is None:
        return jsonify({'error':'Model not ready'}), 500
    proba = clf.predict_proba([emb])[0]
    pred_idx = int(np.argmax(proba))
    confidence = float(proba[pred_idx])
    label = label_encoder.inverse_transform([pred_idx])[0] if label_encoder is not None else str(pred_idx)
    if confidence < UNKNOWN_THRESHOLD:
        label = "unknown"
    return jsonify({'label': label, 'confidence': float(confidence)})



@app.route('/api/verify', methods=['POST'])
def api_verify():
    load_artifacts()
    # expects two files 'image_a' and 'image_b'
    if 'image_a' not in request.files or 'image_b' not in request.files:
        return jsonify({'error':'Provide image_a and image_b files'}), 400
    img_a = Image.open(request.files['image_a'].stream).convert('RGB')
    img_b = Image.open(request.files['image_b'].stream).convert('RGB')
    fa = extract_aligned_face(img_a, mtcnn=get_mtcnn())
    fb = extract_aligned_face(img_b, mtcnn=get_mtcnn())
    if fa is None or fb is None:
        return jsonify({'error':'Face not detected in one of the images'}), 400
    ea = get_embedding(fa, resnet=get_resnet())
    eb = get_embedding(fb, resnet=get_resnet())
    score = float(np.dot(ea, eb))  # cosine with normalized embeddings
    match = score >= COSINE_THRESHOLD
    return jsonify({'similarity': score, 'match': bool(match), 'threshold': COSINE_THRESHOLD})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    # set global paths
    MODEL_DIR = args.model_dir
    CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'classifier.joblib')
    LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.joblib')
    app.run(host=args.host, port=args.port, debug=True)

