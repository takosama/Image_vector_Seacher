"""Non-pickle datasets and non-destructive search exports."""

import math
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PureWindowsPath

import numpy as np

MAX_BYTES = 512 * 1024 * 1024


def _name(value):
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError("image names must be nonempty filenames")
    if "/" in value or "\\" in value or ":" in value or "\x00" in value:
        raise ValueError("image names must not contain paths")
    if PureWindowsPath(value).is_reserved() or value.endswith((" ", ".")):
        raise ValueError("invalid image filename")
    if Path(value).suffix.lower() != ".png":
        raise ValueError("only PNG filenames are supported")
    return value


def _validate(names, vectors):
    names = np.asarray(names)
    vectors = np.asarray(vectors)
    if names.ndim != 1 or names.dtype.kind != "U":
        raise ValueError("filenames must be a Unicode string array")
    clean = [_name(n) for n in names.tolist()]
    if len(set(n.casefold() for n in clean)) != len(clean):
        raise ValueError("duplicate image names")
    if vectors.ndim != 2 or vectors.dtype.kind not in "fi":
        raise ValueError("vectors must be a numeric [N,D] matrix")
    if not len(clean) or vectors.shape[0] != len(clean) or vectors.shape[1] == 0:
        raise ValueError("filename/vector count or dimension mismatch")
    if vectors.shape[0] > 100_000 or vectors.shape[1] > 4096:
        raise ValueError("dataset dimensions exceed limits")
    if names.nbytes + vectors.nbytes > MAX_BYTES or not np.isfinite(vectors).all():
        raise ValueError("dataset is too large or has non-finite vectors")
    return names, vectors


def save_dataset(path, names, vectors):
    names, vectors = _validate(names, vectors)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, filenames=names, vectors=vectors)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)  # Only this call's own temporary file.


def load_dataset(path, max_bytes=MAX_BYTES):
    """Reject pickle, object arrays and oversized/malformed NPY payloads."""
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError(
            "Use dataset.npz; rebuild legacy dataset.pkl from original images"
        )
    if path.stat().st_size > max_bytes:
        raise ValueError("dataset exceeds byte limit")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) != 2 or {e.filename for e in entries} != {
            "filenames.npy",
            "vectors.npy",
        }:
            raise ValueError("unexpected dataset fields")
        if sum(e.file_size for e in entries) > max_bytes:
            raise ValueError("expanded dataset exceeds byte limit")
        for entry in entries:
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("unsupported array version")
                if dtype.hasobject or dtype.kind not in "Ufi":
                    raise ValueError("object or unsupported array dtype")
                expected = math.prod(shape) * dtype.itemsize
                if expected > max_bytes or expected != entry.file_size - stream.tell():
                    raise ValueError("invalid array size")
    with np.load(path, allow_pickle=False) as data:
        names, vectors = _validate(data["filenames"], data["vectors"])
        return dict(zip(names.tolist(), vectors))


def export_matches(names, scores, image_root="img", output_root="similar_images"):
    """Create a new result directory every time; never remove prior contents."""
    names = [_name(n) for n in names]
    scores = list(scores)
    if len(names) != len(scores) or not all(math.isfinite(float(s)) for s in scores):
        raise ValueError("invalid similarity scores")
    source_root = Path(image_root).resolve(strict=True)
    sources = []
    for name in names:
        source = (source_root / name).resolve(strict=True)
        if not source.is_relative_to(source_root) or not source.is_file():
            raise ValueError("image path escapes source directory")
        sources.append(source)
    root = Path(output_root)
    if root.is_symlink():
        raise ValueError("output root must not be a symbolic link")
    root.mkdir(parents=True, exist_ok=True)
    result = Path(tempfile.mkdtemp(prefix="search_", dir=root))
    for rank, (source, score) in enumerate(zip(sources, scores), start=1):
        # Rank prevents rounding/name collisions. Partial failures retain all files.
        destination = result / f"{rank:02d}_{float(score):.5f}_{source.name}"
        shutil.copy2(source, destination)
    return result
