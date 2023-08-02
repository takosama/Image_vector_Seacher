from sklearn.metrics.pairwise import cosine_similarity
import shutil
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from tqdm import tqdm
# pickleファイルから辞書を読み込む
with open('dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

# ベクトルデータを取り出し、配列に変換
vectors = np.array(list(dataset.values()))
if vectors.ndim == 3:
    vectors = vectors.reshape(vectors.shape[0], -1)
# PCAを使用して2次元に投影
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# 2次元のベクトルデータと画像のパスを辞書に保存
dataset_2d = {path: vec for path, vec in zip(dataset.keys(), vectors_2d)}


# 散布図と画像表示エリアのサブプロットを作成
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
x_coords = [vec[0] for vec in dataset_2d.values()]
y_coords = [vec[1] for vec in dataset_2d.values()]

# 散布図をプロット
ax1.scatter(x_coords, y_coords)
ax1.set_xlabel('Principal Component 1')
ax1.set_ylabel('Principal Component 2')
ax1.set_title('2D Projection of Image Vectors')

# 画像表示エリアの初期設定
ax2.axis('off')
image_display = ax2.imshow(np.zeros((800, 800, 3)))  # 初期の空白画像

# クリックイベントの処理
current_image_path = None
zoom_level = 1.0


def on_motion(event):
    global current_image_path
    # イベントが散布図でなければ無視
    if event.inaxes != ax1:
        return

    # イベント位置の最も近い点のインデックスを取得
    distances = [(event.xdata - x)**2 + (event.ydata - y)
                 ** 2 for x, y in zip(x_coords, y_coords)]
    closest_index = np.argmin(distances)

    # 対応する画像のパスを取得
    image_path = list(dataset_2d.keys())[closest_index]

    # 画像が変わっていなければ何もしない
    if image_path == current_image_path:
        return

    current_image_path = image_path

    # 画像を読み込み
    img = Image.open("img/" + image_path)
    img = img.resize((800, 800))  # サイズ調整

    # 画像表示エリアに画像をセット
    image_display.set_data(img)

    # 拡大・縮小の中心をリセット
    ax2.set_xlim([0, 800 * zoom_level])
    ax2.set_ylim([800 * zoom_level, 0])

    plt.draw()


def on_scroll(event):
    # イベントが散布図でなければ無視
    if event.inaxes != ax1:
        return

    # スクロール方向に応じて拡大・縮小
    zoom_factor = 1.1 if event.button == 'up' else 0.9

    # スクロールした点を中心に拡大・縮小
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    xdata = event.xdata
    ydata = event.ydata
    ax1.set_xlim([xdata - (xdata - xlim[0]) * zoom_factor,
                  xdata + (xlim[1] - xdata) * zoom_factor])
    ax1.set_ylim([ydata - (ydata - ylim[0]) * zoom_factor,
                  ydata + (ylim[1] - ydata) * zoom_factor])

    plt.draw()


def on_click(event):
    global current_image_path
    # クリックされたのが散布図でなければ無視
    if event.inaxes != ax1:
        return

    # クリックされた点のインデックスを取得
    distances = [(event.xdata - x)**2 + (event.ydata - y)
                 ** 2 for x, y in zip(x_coords, y_coords)]
    closest_index = np.argmin(distances)

    # 対応する画像のパスとベクトルを取得
    clicked_image_path = list(dataset.keys())[closest_index]
    clicked_vector = vectors[closest_index].reshape(1, -1)

    # コサイン類似度を計算
    similarities = cosine_similarity(clicked_vector, vectors)

    # 類似度でソートし、上位20枚のインデックスを取得
    top_indices = np.argsort(similarities[0])[-20:][::-1]

    # 新しいフォルダを作成
    folder_name = "similar_images"
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)

    # 上位20枚の画像をフォルダに保存
    for index in top_indices:
        image_path = list(dataset.keys())[index]
        similarity_score = similarities[0][index]
        img = Image.open("img/" + image_path)
        new_filename = f"{similarity_score:.5f}_{os.path.basename(image_path)}"
        img.save(os.path.join(folder_name, new_filename))
    # 画像フォルダを表示（オプション）
    os.startfile(folder_name)

    # 以前のマウスモーションイベントの処理を続行
    on_motion(event)


# イベントリスナーを追加
plt.gcf().canvas.mpl_connect('button_press_event', on_click)
# イベントリスナーを追加
plt.gcf().canvas.mpl_connect('motion_notify_event', on_motion)
plt.gcf().canvas.mpl_connect('scroll_event', on_scroll)

# プロットの表示
plt.show()
