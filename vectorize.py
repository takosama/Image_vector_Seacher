from tqdm import tqdm
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from clip import clip
import japanese_clip as ja_clip
import pickle
import os
# モデルと前処理のロード
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = ja_clip.load(
    "rinna/japanese-cloob-vit-b-16", device=device)
tokenizer = ja_clip.load_tokenizer()

# ファイル名と特徴ベクトルを保存する辞書
dataset = {}

# imgフォルダ内のすべての画像ファイルのパスを取得
image_paths = [os.path.join("img", filename) for filename in os.listdir(
    "img") if filename.endswith(".png")]
for image_path in tqdm(image_paths):
    # 画像のロードと前処理
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# 画像から特徴ベクトルを取得
    image_features = model.encode_image(image)
    # ファイル名を取得（拡張子あり）
    filename_with_extension = os.path.basename(image_path)

    # 辞書にファイル名と特徴ベクトルを保存
    dataset[filename_with_extension] = image_features.cpu().numpy()

# 辞書をpickleファイルとして保存
with open('dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)


# 特徴ベクトルはキャプションベクトルとして使用できます
text = "A description of the image."


a = torch.tensor([tokenizer.encode(text)]) .to(device)
# テキストから特徴ベクトルを取得
text_features = model.encode_text(a)

print(text_features)
