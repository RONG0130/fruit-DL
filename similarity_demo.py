"""僅供本機 Streamlit demo 使用的圖片特徵與 FAISS 檢索。"""

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class SimilarImage:
    path: Path
    similarity: float


def checkpoint_sha256(path: Path) -> str:
    """用權重內容辨識索引版本，避免同名 run 的檔案被換掉。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype="float32").reshape(-1)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("圖片特徵向量無效")
    return np.ascontiguousarray(vector / norm, dtype="float32")


def predict_with_embedding(predictor, image: Image.Image) -> tuple[dict[str, float], np.ndarray]:
    """同一次 DINOv3 前向運算取得完整類別機率與分類前特徵。"""
    tensor = predictor.transform(image.convert("RGB")).unsqueeze(0).to(predictor.device)
    with torch.inference_mode():
        tokens = predictor.model.forward_features(tensor)
        features = predictor.model.forward_head(tokens, pre_logits=True)
        logits = predictor.model.forward_head(tokens)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    scores = {
        predictor.class_names[index]: float(probability)
        for index, probability in enumerate(probabilities)
    }
    embedding = _unit_vector(features[0].cpu().numpy())
    return scores, embedding


def select_representative(
    samples: list[tuple[dict[str, float], np.ndarray]], class_name: str
) -> tuple[int, np.ndarray]:
    """選該類分數最高的上傳照片；同分時保留最早上傳的一張。"""
    if not samples or any(class_name not in scores for scores, _ in samples):
        raise ValueError(f"缺少 {class_name} 的逐張預測分數")
    index = max(range(len(samples)), key=lambda number: samples[number][0][class_name])
    return index, samples[index][1]


def build_crop_indexes(
    plant: str,
    data_dir: Path,
    output_dir: Path,
    class_names: list[str],
    model_sha256: str,
    embed_image: Callable[[Path], np.ndarray],
) -> dict[str, int]:
    """以 train 原圖建立一個作物資料庫，內含每病徵的獨立索引。"""
    import faiss

    data_dir = data_dir.resolve()
    train_dir = data_dir / "train"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "plant": plant,
        "data_dir": str(data_dir),
        "model_sha256": model_sha256,
        "classes": {},
    }
    counts = {}
    for class_name in class_names:
        if Path(class_name).name != class_name:
            raise ValueError(f"類別名稱不可作為資料夾：{class_name}")
        folder = train_dir / class_name
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        paths = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        vectors = []
        relative_paths = []
        seen_hashes = set()
        for path in paths:
            image_hash = hashlib.sha256(path.read_bytes()).digest()
            if image_hash in seen_hashes:
                continue
            with Image.open(path) as image:
                image.load()
                vector = _unit_vector(embed_image(path))
            seen_hashes.add(image_hash)
            vectors.append(vector)
            relative_paths.append(path.relative_to(data_dir).as_posix())
        if not vectors:
            raise ValueError(f"{plant}/{class_name} 沒有可建立索引的 train 圖片")
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise ValueError(f"{plant}/{class_name} 的特徵維度不一致")
        index = faiss.IndexFlatIP(len(vectors[0]))
        index.add(np.stack(vectors))
        # FAISS 的 Windows 檔名介面不支援此專案路徑中的中文字；由 Python 寫入位元組。
        (output_dir / f"{class_name}.faiss").write_bytes(faiss.serialize_index(index).tobytes())
        manifest["classes"][class_name] = relative_paths
        counts[class_name] = len(relative_paths)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return counts


def search_similar(
    index_dir: Path,
    plant: str,
    class_name: str,
    embedding: np.ndarray,
    model_sha256: str,
    top_k: int = 3,
) -> list[SimilarImage]:
    """只在指定作物與病徵的 train 參考照片內搜尋。"""
    import faiss

    if top_k < 1:
        raise ValueError("top_k 必須大於零")
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["plant"] != plant:
        raise ValueError("索引作物與目前診斷不符")
    if manifest["model_sha256"] != model_sha256:
        raise ValueError("索引模型版本與目前診斷模型不符，請重建索引")
    if class_name not in manifest["classes"]:
        raise ValueError(f"索引沒有 {class_name} 類別")
    image_paths = manifest["classes"][class_name]
    index_bytes = (index_dir / f"{class_name}.faiss").read_bytes()
    index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype="uint8").copy())
    query = _unit_vector(embedding)
    if len(query) != index.d or index.ntotal != len(image_paths):
        raise ValueError("索引維度或圖片清單不一致，請重建索引")
    scores, ids = index.search(query.reshape(1, -1), index.ntotal)
    data_dir = Path(manifest["data_dir"]).resolve()
    hits = []
    for score, image_id in zip(scores[0], ids[0]):
        if image_id < 0:
            continue
        path = (data_dir / image_paths[int(image_id)]).resolve()
        if not path.is_relative_to(data_dir / "train") or not path.is_file():
            continue
        hits.append(SimilarImage(path=path, similarity=float(score)))
        if len(hits) == top_k:
            break
    return hits


def retrieve_demo_gallery(
    plant: str,
    result: dict,
    images: list[Image.Image],
    holder,
    index_root: Path,
) -> tuple[int, list[SimilarImage]] | None:
    """沿用 API 的整批第一名，選其逐張最高分圖片並查同類參考照。"""
    if result.get("need_more_info"):
        return None
    class_name = result.get("raw_output", {}).get("model_label")
    if not class_name:
        raise ValueError("API 結果缺少原始類別標籤")
    if not images:
        raise ValueError("沒有上傳圖片可供檢索")
    manifest_path = index_root / plant / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到 {plant} 的 FAISS 索引，請先建立索引")
    with holder.lock:
        predictor, model_sha256 = holder.get(plant)
        if class_name not in predictor.class_names:
            raise ValueError("API 預測類別與本機模型不一致")
        samples = [predict_with_embedding(predictor, image) for image in images]
        selected_index, embedding = select_representative(samples, class_name)
        hits = search_similar(
            index_dir=index_root / plant,
            plant=plant,
            class_name=class_name,
            embedding=embedding,
            model_sha256=model_sha256,
            top_k=3,
        )
    return selected_index, hits


class DemoPredictorHolder:
    """Streamlit 程序只在 CPU 保留一個作物模型。"""

    def __init__(self):
        self.lock = threading.RLock()
        self.plant = None
        self.predictor = None
        self.model_sha256 = None

    def get(self, plant: str):
        from api.model_registry import MODEL_SPECS, check_artifacts
        from predict import PlantDiseasePredictor

        with self.lock:
            if self.plant == plant and self.predictor is not None:
                return self.predictor, self.model_sha256
            check_artifacts(plant)
            spec = MODEL_SPECS[plant]
            self.predictor = None
            self.plant = None
            predictor = PlantDiseasePredictor(
                model_path=str(spec.model_path),
                classes_path=str(spec.classes_path),
                device="cpu",
                verbose=False,
            )
            self.predictor = predictor
            self.model_sha256 = checkpoint_sha256(spec.model_path)
            self.plant = plant
            return predictor, self.model_sha256
