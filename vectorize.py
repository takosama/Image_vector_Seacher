"""Build a non-pickle feature dataset from local PNG images."""
from pathlib import Path
from storage import save_dataset


def main():
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    import japanese_clip as ja_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = ja_clip.load("rinna/japanese-cloob-vit-b-16", device=device)
    model.eval()
    names, vectors = [], []
    with torch.inference_mode():
        for path in tqdm(sorted(Path("img").glob("*.png"))):
            with Image.open(path) as image:
                tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
            vectors.append(model.encode_image(tensor).float().cpu().numpy().reshape(-1))
            names.append(path.name)
    if not names:
        raise ValueError("No PNG images found in img/")
    save_dataset("dataset.npz", np.asarray(names), np.stack(vectors))


if __name__ == "__main__":
    main()
