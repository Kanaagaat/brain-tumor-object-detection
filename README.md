# Brain Tumor Detection from MRI

A CSCI1409 bonus project exploring object detection on brain MRI images. The current notebook audits and visualizes YOLO annotations and prepares a preliminary cleaned copy of the data. Model training, a controlled experiment, and final evaluation are in progress.

[Open the notebook in Google Colab](https://colab.research.google.com/github/Kanaagaat/brain-tumor-object-detection/blob/main/notebook.ipynb)

## Dataset

The [MRI for Brain Tumor with Bounding Boxes dataset](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes) contains 5,249 MRI images in axial, sagittal, and coronal views. The dataset page lists its license as CC0. Bounding boxes use the YOLO format: class ID followed by normalized center coordinates, width, and height.

| Original category | Class ID | Train images | Validation images |
| --- | ---: | ---: | ---: |
| Glioma | 0 | 1,153 | 136 |
| Meningioma | 1 | 1,449 | 140 |
| No Tumor | 2 | 711 | 100 |
| Pituitary | 3 | 1,424 | 136 |
| Total | | 4,737 | 512 |

The original dataset does not provide a separate test split. The notebook documents issues found during annotation inspection, including boxes for the No Tumor category, mismatches between folder names and box classes, and missing or empty labels. Its preliminary cleaned copy preserves the original class IDs, including Pituitary as 3. Further review and a train/validation/test split remain necessary.

## Explore the data

1. Download the dataset from its [Kaggle page](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes) and extract it into `data/archive/`, so that `data/archive/Train/` and `data/archive/Val/` exist.
2. Open [notebook.ipynb](notebook.ipynb) in Jupyter or Colab. Install the imports used in the notebook, including NumPy, PyTorch, pandas, Matplotlib, and Pillow, in your environment.
3. Run the notebook cells from the project root. The audit writes reports under `reports/` and creates a separate preliminary copy in `data/clean_v1/`.

The Colab notebook currently requires the dataset to be made available at the path above and is still being developed. It does not yet provide a complete fresh-runtime training and evaluation workflow.

To regenerate the original annotation figures locally, install Matplotlib and Pillow and run:

```bash
python visualize_labels.py --root data/archive --out viz --seed 42
```

## Repository contents

- `notebook.ipynb` contains the dataset overview, visual review, audit, and initial cleaning work.
- `visualize_labels.py` generates annotation inspection figures.
- `viz/` contains examples from the annotation review.
- `reports/` contains audit and preliminary cleaning manifests.

The source and cleaned image datasets are excluded from Git. Training runs, evaluation metrics, and model weights will be added as the project progresses.

## Limitations

Annotation quality and potential similarity between images require further review. Patient-level independence cannot be assumed without reliable patient identifiers. This is an educational project.
