"""Demo 相似圖片索引與查詢的行為測試。"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from similarity_demo import (
    build_crop_indexes,
    retrieve_demo_gallery,
    search_similar,
    select_representative,
)


def _photo(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def test_representative_uses_score_of_group_winner_even_when_not_image_top_one():
    samples = [
        ({"Rust": 0.40, "Aphids": 0.60}, np.array([1, 0], dtype="float32")),
        ({"Rust": 0.75, "Aphids": 0.25}, np.array([0, 1], dtype="float32")),
    ]

    index, embedding = select_representative(samples, "Rust")

    assert index == 1
    np.testing.assert_array_equal(embedding, [0, 1])


def test_build_and_search_stay_within_crop_and_class_and_use_train_only(tmp_path):
    data = tmp_path / "datasets" / "annona" / "v1"
    _photo(data / "train" / "Rust" / "near.png", "red")
    _photo(data / "train" / "Rust" / "near_copy.png", "red")
    _photo(data / "train" / "Rust" / "far.png", "green")
    _photo(data / "train" / "Anthracnose" / "other.png", "blue")
    _photo(data / "test" / "Rust" / "held_out.png", "yellow")
    vectors = {
        "near.png": [1, 0],
        "near_copy.png": [1, 0],
        "far.png": [0, 1],
        "other.png": [1, 0],
    }

    build_crop_indexes(
        plant="annona",
        data_dir=data,
        output_dir=tmp_path / "indexes" / "annona",
        class_names=["Rust", "Anthracnose"],
        model_sha256="abc123",
        embed_image=lambda path: np.array(vectors[path.name], dtype="float32"),
    )
    hits = search_similar(
        index_dir=tmp_path / "indexes" / "annona",
        plant="annona",
        class_name="Rust",
        embedding=np.array([10, 0], dtype="float32"),
        model_sha256="abc123",
        top_k=3,
    )

    assert [hit.path.name for hit in hits] == ["near.png", "far.png"]
    assert all(hit.path.parent.name == "Rust" for hit in hits)
    assert hits[0].similarity == pytest.approx(1.0)
    with pytest.raises(ValueError, match="作物"):
        search_similar(
            index_dir=tmp_path / "indexes" / "annona",
            plant="tea",
            class_name="Rust",
            embedding=np.array([1, 0], dtype="float32"),
            model_sha256="abc123",
        )


def test_search_rejects_model_version_mismatch(tmp_path):
    data = tmp_path / "data"
    _photo(data / "train" / "Rust" / "sample.png", "red")
    index_dir = tmp_path / "index"
    build_crop_indexes(
        plant="annona",
        data_dir=data,
        output_dir=index_dir,
        class_names=["Rust"],
        model_sha256="old-model",
        embed_image=lambda _: np.array([1, 0], dtype="float32"),
    )

    with pytest.raises(ValueError, match="模型版本"):
        search_similar(
            index_dir=index_dir,
            plant="annona",
            class_name="Rust",
            embedding=np.array([1, 0], dtype="float32"),
            model_sha256="new-model",
        )


def test_gallery_chooses_image_with_highest_score_for_api_winner(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _photo(data / "train" / "Rust" / "reference.png", "red")
    index_dir = tmp_path / "indexes" / "annona"
    build_crop_indexes(
        plant="annona",
        data_dir=data,
        output_dir=index_dir,
        class_names=["Rust"],
        model_sha256="same-model",
        embed_image=lambda _: np.array([0, 1], dtype="float32"),
    )

    class TrackingLock:
        held = False

        def __enter__(self):
            self.held = True

        def __exit__(self, *_):
            self.held = False

    class Holder:
        lock = TrackingLock()

        def get(self, plant):
            assert plant == "annona"
            assert self.lock.held
            return type("Predictor", (), {"class_names": ["Rust"]})(), "same-model"

    def fake_predict(_predictor, image):
        assert holder.lock.held
        if image.getpixel((0, 0)) == (255, 0, 0):
            return {"Rust": 0.2}, np.array([1, 0], dtype="float32")
        return {"Rust": 0.9}, np.array([0, 1], dtype="float32")

    holder = Holder()
    monkeypatch.setattr("similarity_demo.predict_with_embedding", fake_predict)
    selected, hits = retrieve_demo_gallery(
        plant="annona",
        result={"raw_output": {"model_label": "Rust"}, "need_more_info": []},
        images=[Image.new("RGB", (8, 8), "red"), Image.new("RGB", (8, 8), "blue")],
        holder=holder,
        index_root=tmp_path / "indexes",
    )

    assert selected == 1
    assert [hit.path.name for hit in hits] == ["reference.png"]


def test_gallery_skips_uncertain_api_result(tmp_path):
    assert (
        retrieve_demo_gallery(
            plant="annona",
            result={"raw_output": {"model_label": "Rust"}, "need_more_info": [{"type": "image"}]},
            images=[Image.new("RGB", (8, 8), "red")],
            holder=None,
            index_root=tmp_path,
        )
        is None
    )


def test_gallery_missing_index_does_not_load_large_model(tmp_path):
    class Holder:
        def get(self, _plant):
            raise AssertionError("missing index must be found before model load")

    with pytest.raises(FileNotFoundError):
        retrieve_demo_gallery(
            plant="annona",
            result={"raw_output": {"model_label": "Rust"}, "need_more_info": []},
            images=[Image.new("RGB", (8, 8), "red")],
            holder=Holder(),
            index_root=tmp_path,
        )
