"""為 Streamlit demo 離線建立四作物的相似圖片索引。"""

import argparse
import gc
from pathlib import Path

import torch

from api.model_registry import MODEL_SPECS, check_artifacts
from predict import PlantDiseasePredictor
from similarity_demo import build_crop_indexes, checkpoint_sha256, predict_with_embedding


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="建立 Streamlit demo 的 FAISS 圖片索引")
    parser.add_argument("--plant", choices=[*MODEL_SPECS, "all"], required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runtime" / "faiss")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    plants = list(MODEL_SPECS) if args.plant == "all" else [args.plant]
    for plant in plants:
        check_artifacts(plant)
        spec = MODEL_SPECS[plant]
        predictor = PlantDiseasePredictor(
            model_path=str(spec.model_path),
            classes_path=str(spec.classes_path),
            device=args.device,
            verbose=False,
        )

        def embed_image(path: Path):
            from PIL import Image

            with Image.open(path) as image:
                _, embedding = predict_with_embedding(predictor, image)
            return embedding

        counts = build_crop_indexes(
            plant=plant,
            data_dir=args.data_root / plant / args.version,
            output_dir=args.output_root / plant,
            class_names=predictor.class_names,
            model_sha256=checkpoint_sha256(spec.model_path),
            embed_image=embed_image,
        )
        print(f"{plant}: " + ", ".join(f"{name}={count}" for name, count in counts.items()))
        del predictor
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
