"""
Utilities: detection, alignment, embedding extraction and helper IO functions.

Uses facenet-pytorch for MTCNN and InceptionResnetV1.
"""
import os
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2

# Create global devices and models lazily to speed scripts importing utils
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_mtcnn = None
_resnet = None

def get_mtcnn(keep_all=False, device=None, post_process=True, margin=0):
    global _mtcnn
    if _mtcnn is None:
        device = device or _device
        _mtcnn = MTCNN(image_size=160, margin=margin, keep_all=keep_all, device=device, post_process=post_process)
    return _mtcnn

def get_resnet(pretrained='vggface2', device=None):
    global _resnet
    if _resnet is None:
        device = device or _device
        _resnet = InceptionResnetV1(pretrained=pretrained).to(device).eval()
    return _resnet

def read_image(img_path):
    """Read an image and return RGB PIL image."""
    img = Image.open(img_path).convert('RGB')
    return img

def extract_aligned_face(img, mtcnn=None, device=None):
    """
    Accepts PIL image or numpy array, returns aligned face tensor (1,3,160,160) or None.
    If multiple faces are detected and keep_all=False in MTCNN, returns the best face.
    """
    if mtcnn is None:
        mtcnn = get_mtcnn()
    # mtcnn expects PIL Image or torch tensor
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    face = mtcnn(img)
    # mtcnn returns torch.Tensor of shape (3,160,160) or a list if keep_all True
    if isinstance(face, list):
        if len(face) == 0:
            return None
        # take the first face
        face = face[0]
    if face is None:
        return None
    return face.unsqueeze(0)  # add batch dim

def get_embedding(face_tensor, resnet=None, device=None):
    """
    Accepts face_tensor of shape (1,3,160,160) in float; returns 512-d numpy embedding (L2-normalized).
    """
    if face_tensor is None:
        return None
    if resnet is None:
        resnet = get_resnet()
    device = device or _device
    with torch.no_grad():
        face_tensor = face_tensor.to(device)
        emb = resnet(face_tensor)  # shape (1,512)
        emb_np = emb.cpu().numpy().reshape(-1)
        # L2 normalize
        emb_np = emb_np / np.linalg.norm(emb_np)
        return emb_np

def image_to_embedding(img_path, mtcnn=None, resnet=None):
    """
    Full pipeline: read image, detect & align face, get embedding.
    Returns embedding (512,) or None if no face found.
    """
    try:
        img = read_image(img_path)
    except Exception:
        return None
    if mtcnn is None:
        mtcnn = get_mtcnn()
    if resnet is None:
        resnet = get_resnet()
    face = extract_aligned_face(img, mtcnn=mtcnn)
    if face is None:
        return None
    emb = get_embedding(face, resnet=resnet)
    return emb

def batch_extract_embeddings(data_dir, out_path, ext=('.jpg','.jpeg','.png'), verbose=True):
    """
    Walk data_dir which has subfolders per identity:
      data_dir/PersonA/*.jpg
      data_dir/PersonB/*.jpg
    Produces saved npz with:
      embeddings: (N,512)
      labels: list N strings
      paths: list of N image paths
    """
    mtcnn = get_mtcnn()
    resnet = get_resnet()
    emb_list = []
    labels = []
    paths = []

    for person in sorted(os.listdir(data_dir)):
        p_path = os.path.join(data_dir, person)
        if not os.path.isdir(p_path):
            continue
        for fname in sorted(os.listdir(p_path)):
            if not fname.lower().endswith(ext):
                continue
            img_path = os.path.join(p_path, fname)
            emb = image_to_embedding(img_path, mtcnn=mtcnn, resnet=resnet)
            if emb is None:
                if verbose:
                    print(f"[WARN] No face detected in {img_path}; skipping")
                continue
            emb_list.append(emb)
            labels.append(person)
            paths.append(img_path)
    if len(emb_list) == 0:
        raise RuntimeError("No embeddings extracted. Check data_dir and image files.")
    embeddings = np.vstack(emb_list)
    np.savez_compressed(out_path, embeddings=embeddings, labels=np.array(labels), paths=np.array(paths))
    if verbose:
        print(f"[INFO] Saved {embeddings.shape[0]} embeddings to {out_path}")
    return out_path

def load_embeddings(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data['embeddings']
    labels = data['labels']
    paths = data['paths']
    return embeddings, labels, paths

def cosine_similarity(a, b):
    # a: (d,), b: (d,) or (N,d)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.dot(b, a) if b.ndim==2 else np.dot(a, b)

if __name__ == "__main__":
    # small test: run extraction on a tiny folder if desired
    pass