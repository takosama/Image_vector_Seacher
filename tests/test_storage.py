import numpy as np
import pytest

import storage


def test_round_trip(tmp_path):
    path = tmp_path / "dataset.npz"
    storage.save_dataset(path, ["a.png", "日本.png"], np.arange(6.0).reshape(2, 3))
    loaded = storage.load_dataset(path)
    assert list(loaded) == ["a.png", "日本.png"]
    np.testing.assert_array_equal(loaded["日本.png"], [3, 4, 5])


@pytest.mark.parametrize(
    "name", ["../a.png", "/a.png", "C:\\a.png", "a/b.png", "NUL.png"]
)
def test_path_rejection(tmp_path, name):
    with pytest.raises(ValueError):
        storage.save_dataset(tmp_path / "d.npz", [name], [[1.0]])


def test_legacy_and_object_rejected(tmp_path):
    with pytest.raises(ValueError, match="rebuild"):
        storage.load_dataset(tmp_path / "dataset.pkl")
    path = tmp_path / "d.npz"
    np.savez(path, filenames=np.array(["a.png"], dtype=object), vectors=np.ones((1, 2)))
    with pytest.raises(ValueError, match="object"):
        storage.load_dataset(path)


@pytest.mark.parametrize(
    "vectors", [np.array([[np.nan]]), np.ones((2, 1)), np.ones((1, 1, 1))]
)
def test_invalid_vectors(tmp_path, vectors):
    with pytest.raises(ValueError):
        storage.save_dataset(tmp_path / "d.npz", ["a.png"], vectors)


def test_export_never_deletes_old_or_unrelated_files(tmp_path, monkeypatch):
    images = tmp_path / "img"
    images.mkdir()
    (images / "a.png").write_bytes(b"fixture")
    out = tmp_path / "similar_images"
    out.mkdir()
    sentinel = out / "notes.txt"
    sentinel.write_text("keep")
    a = storage.export_matches(["a.png"], [1.0], images, out)
    b = storage.export_matches(["a.png"], [1.0], images, out)
    assert a != b and len(list(a.iterdir())) == len(list(b.iterdir())) == 1
    monkeypatch.setattr(
        storage.shutil,
        "copy2",
        lambda *args: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError):
        storage.export_matches(["a.png"], [1.0], images, out)
    assert sentinel.read_text() == "keep" and a.exists() and b.exists()


def test_symlink_escape_rejected(tmp_path):
    images = tmp_path / "img"
    images.mkdir()
    (tmp_path / "outside.png").write_bytes(b"x")
    (images / "a.png").symlink_to(tmp_path / "outside.png")
    with pytest.raises(ValueError, match="escapes"):
        storage.export_matches(["a.png"], [1.0], images, tmp_path / "out")


def test_expanded_byte_limit(tmp_path):
    path = tmp_path / "d.npz"
    np.savez_compressed(
        path, filenames=np.array(["a.png"]), vectors=np.zeros((1, 1024))
    )
    with pytest.raises(ValueError, match="byte limit"):
        storage.load_dataset(path, max_bytes=1024)


def test_view_click_integration_has_no_destructive_reset(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseEvent
    from PIL import Image

    import view

    monkeypatch.chdir(tmp_path)
    images = tmp_path / "img"
    images.mkdir()
    for name, color in [("a.png", "red"), ("b.png", "blue")]:
        Image.new("RGB", (2, 2), color).save(images / name)
    storage.save_dataset("dataset.npz", ["a.png", "b.png"], [[1.0, 0.0], [0.0, 1.0]])
    output = tmp_path / "similar_images"
    output.mkdir()
    (output / "notes.txt").write_text("keep")
    monkeypatch.setattr(plt, "show", lambda: None)
    view.main()
    figure = plt.gcf()
    figure.canvas.draw()
    axis = figure.axes[0]
    coords = axis.collections[0].get_offsets()[0]
    x, y = axis.transData.transform(coords)
    event = MouseEvent("button_press_event", figure.canvas, x, y, button=1)
    figure.canvas.callbacks.process("button_press_event", event)
    figure.canvas.callbacks.process("button_press_event", event)
    assert len(list(output.glob("search_*"))) == 2
    assert (output / "notes.txt").read_text() == "keep"
    plt.close(figure)
