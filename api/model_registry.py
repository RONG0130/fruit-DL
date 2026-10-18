"""固定的植物模型登錄表與單 GPU 模型切換。"""

import asyncio
import gc
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from predict import PlantDiseasePredictor


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelSpec:
    run: str
    classes: tuple[str, ...]

    @property
    def model_path(self) -> Path:
        return PROJECT_ROOT / self.run / "best_model.pth"

    @property
    def classes_path(self) -> Path:
        return PROJECT_ROOT / self.run / "classes.json"


# 路徑與類別固定在伺服器端，請求不得指定任意模型檔。
MODEL_SPECS = {
    "annona": ModelSpec("runs/annona/20261006-215715_batch8-lr2e-5", (
        "Algal_leaf_spot", "Anthracnose", "Kanzawa_spider_mite",
        "Mealybug", "Rust", "Sugar_apple_whitefly", "Tea_red_spider_mite",
    )),
    "tea": ModelSpec("runs/tea/20261006-223812_batch8-lr2e-5", (
        "Agromyzidae", "Caloptilia_theivora", "Oligonychus_coffeae", "Tortrix",
    )),
    "legume": ModelSpec("runs/legume/20261006-232420_batch8-lr2e-5", (
        "Liriomyza_spp", "Rust", "bean_aphid",
    )),
    "facility_leafy_greens": ModelSpec(
        "runs/facility_leafy_greens/20261007-000803_batch8-lr2e-5", (
            "Aphids", "Phaedon_brassicae_Baly", "Pieris_rapae",
            "Striped_flea_beetle",
        ),
    ),
}


def check_artifacts(plant: str) -> None:
    """拒絕缺檔及標籤與預期不符的模型。"""
    spec = MODEL_SPECS[plant]
    if not spec.model_path.is_file() or spec.model_path.stat().st_size == 0:
        raise FileNotFoundError(spec.model_path)
    with spec.classes_path.open(encoding="utf-8") as stream:
        labels = json.load(stream)
    actual = tuple(labels[str(index)] for index in range(len(labels)))
    if actual != spec.classes:
        raise ValueError(f"{plant} classes.json 與模型登錄表不符")


def load_plant_predictor(plant: str) -> PlantDiseasePredictor:
    check_artifacts(plant)
    spec = MODEL_SPECS[plant]
    return PlantDiseasePredictor(
        model_path=str(spec.model_path),
        classes_path=str(spec.classes_path),
        verbose=True,
    )


class ModelManager:
    """同時只持有一個 predictor；鎖涵蓋載入與整段推論。"""

    def __init__(self, loader: Callable = load_plant_predictor):
        self.loader = loader
        self.lock = asyncio.Lock()
        self.loaded_plant = None
        self.predictor = None

    def _switch(self, plant: str):
        if self.loaded_plant == plant and self.predictor is not None:
            return self.predictor
        # 先釋放舊 CUDA 模型，再建立下一個；避免四模型常駐 8GB GPU。
        self.predictor = None
        self.loaded_plant = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        self.predictor = self.loader(plant)
        self.loaded_plant = plant
        return self.predictor

    @asynccontextmanager
    async def use(self, plant: str):
        async with self.lock:
            loading = asyncio.create_task(asyncio.to_thread(self._switch, plant))
            try:
                predictor = await asyncio.shield(loading)
            except asyncio.CancelledError:
                # to_thread 不會因 coroutine 取消而停止；等載入結束才放鎖。
                await loading
                raise
            yield predictor
