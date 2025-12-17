```markdown
# Dataset Format & Public Datasets

Folder structure expected for enrollment / training embeddings:

data/
  train/
    alice/
      img001.jpg
      img002.jpg
      ...
    bob/
      img001.jpg
      img002.jpg
      ...
  val/   (optional)
    alice/
      ...
    bob/
      ...

- Each subfolder name is the identity label.
- Images should contain mostly frontal or near-frontal faces; some variation in pose and lighting is fine.
- At least 5–10 images per person recommended for good classifier training; verification works with fewer samples if using template averaging.

Public datasets for prototyping
- LFW (Labelled Faces in the Wild): good for verification experiments (http://vis-www.cs.umass.edu/lfw/)
- VGGFace2: large-scale dataset suitable for training embeddings (if training from scratch)
- MS-Celeb-1M (historical) — not recommended due to licensing/consent issues
- Use synthetic private datasets or collect consented photos for production.

Tips
- Clean images: remove corrupt files, very small images, or images with no faces.
- If dataset is imbalanced (some people have many more images), either downsample or use class-weighted learning, and evaluate carefully.