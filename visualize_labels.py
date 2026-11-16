"""Visualise YOLO ground-truth boxes for the brain-tumour dataset.

Usage:  python visualize_labels.py [--root data/archive] [--out viz] [--seed 42]

Creates PNG grids in --out:
  01_<split>_per_class.png      random samples of every class folder
  02_no_tumor.png               more No Tumor samples (what does class 2 box cover?)
  03_multi_class.png            files whose label contains >1 class id
  04_foreign_class_only.png     files whose label has no box of the folder's own class
  05_empty_and_unmatched.png    empty label file + images without a label file
and prints a text report.
"""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}
FOLDER_TO_ID = {v: k for k, v in CLASS_NAMES.items()}
COLORS = {0: "#ff3b30", 1: "#ffcc00", 2: "#34c759", 3: "#0a84ff"}


def read_labels(path: Path):
    """Return list of (cls, xc, yc, w, h) or None if the file does not exist."""
    if not path.exists():
        return None
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            boxes.append((int(parts[0]), *map(float, parts[1:])))
    return boxes


def collect(root: Path):
    items = []
    for split in ("Train", "Val"):
        for folder in sorted(p for p in (root / split).iterdir() if p.is_dir()):
            for img in sorted((folder / "images").glob("*")):
                if img.name.startswith("."):
                    continue
                lbl = folder / "labels" / (img.stem + ".txt")
                items.append(dict(split=split, folder=folder.name, img=img, lbl=lbl,
                                  boxes=read_labels(lbl)))
    return items


def draw(ax, item, show_folder=True):
    im = Image.open(item["img"]).convert("RGB")
    W, H = im.size
    ax.imshow(im)
    ax.set_xticks([]); ax.set_yticks([])
    boxes = item["boxes"]
    for c, xc, yc, w, h in boxes or []:
        x0, y0 = (xc - w / 2) * W, (yc - h / 2) * H
        col = COLORS.get(c, "white")
        ax.add_patch(Rectangle((x0, y0), w * W, h * H, fill=False, ec=col, lw=2))
        ax.text(x0, max(y0 - 3, 0), CLASS_NAMES.get(c, f"id {c}"), color="black", fontsize=7,
                va="bottom", bbox=dict(fc=col, ec="none", pad=1))
    status = ("NO LABEL FILE" if boxes is None else "EMPTY LABEL" if not boxes
              else f"{len(boxes)} box" + ("es" if len(boxes) > 1 else ""))
    title = f"{item['img'].name}\n"
    if show_folder:
        title += f"{item['split']}/{item['folder']} · "
    ax.set_title(title + status, fontsize=7)


def grid(items, path, title, cols=4):
    if not items:
        return
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.5))
    axes = [axes] if rows * cols == 1 else list(getattr(axes, "flat", axes))
    for ax, it in zip(axes, items):
        draw(ax, it)
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/archive")
    ap.add_argument("--out", default="viz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-class", type=int, default=4)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = collect(root)

    # 1. random samples per class, per split
    for split in ("Train", "Val"):
        sel = []
        for name in CLASS_NAMES.values():
            pool = [i for i in items if i["split"] == split and i["folder"] == name and i["boxes"]]
            sel += rng.sample(pool, min(a.per_class, len(pool)))
        grid(sel, out / f"01_{split.lower()}_per_class.png",
             f"{split}: {a.per_class} random images per class (seed {a.seed})", cols=a.per_class)

    # 2. No Tumor
    nt = [i for i in items if i["folder"] == "No Tumor" and i["boxes"]]
    grid(rng.sample(nt, min(12, len(nt))), out / "02_no_tumor.png",
         "No Tumor: what does the class-2 box cover?")
    areas = sorted(b[3] * b[4] for i in nt for b in i["boxes"])
    print(f"\nNo Tumor: {len(nt)} files, box area (fraction of image) "
          f"min={areas[0]:.2f} median={areas[len(areas)//2]:.2f} max={areas[-1]:.2f}")

    # 3. multi-class
    multi = [i for i in items if i["boxes"] and len({b[0] for b in i["boxes"]}) > 1]
    print(f"Multi-class files: {len(multi)}")
    grid(rng.sample(multi, min(12, len(multi))), out / "03_multi_class.png",
         f"Label files with >1 class ({len(multi)} total, 12 shown)")

    # 4. no box of the folder's own class
    foreign = [i for i in items if i["boxes"]
               and FOLDER_TO_ID[i["folder"]] not in {b[0] for b in i["boxes"]}]
    print(f"Files without a box of their folder's class: {len(foreign)}")
    for i in foreign:
        print(f"  {i['split']}/{i['folder']}/{i['img'].name}: "
              f"{[CLASS_NAMES[b[0]] for b in i['boxes']]}")
    grid(foreign, out / "04_foreign_class_only.png",
         "Folder class ≠ label class (possible wrong labels)")

    # 5. empty labels & images without label file
    bad = [i for i in items if not i["boxes"]]
    for i in bad:
        print(f"  {'NO LABEL FILE' if i['boxes'] is None else 'EMPTY'}: "
              f"{i['split']}/{i['folder']}/{i['img'].name}")
    orphan = [l for l in root.glob("*/*/labels/*.txt")
              if not any((l.parent.parent / "images").glob(l.stem + ".*"))]
    for l in orphan:
        print(f"  LABEL WITHOUT IMAGE: {l.relative_to(root)} -> {l.read_text().strip()}")
    grid(bad, out / "05_empty_and_unmatched.png", "Empty label file / image without label",
         cols=max(1, len(bad)))


if __name__ == "__main__":
    main()
