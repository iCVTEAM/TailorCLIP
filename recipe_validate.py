from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

import chinopie
from chinopie import EvaluationRecipe, TrainBootstrap, logger
from chinopie.modelhelper import HyperparameterManager, ModelStaff
from chinopie.probes import AveragePrecisionMeter

import clip_nopooling
import utils
from models.adaptive_clip import AdaptiveClipDecodeStyle
from models.estimator import Estimator
from recipe_interpolation import (
    AdaptiveClipWithEstimator,
    SUPPORTED_BACKBONES,
    get_dataset,
    transform_train_hard,
    transform_zeroshot,
)


class ValidateRecipe(EvaluationRecipe):
    def __init__(self, trainset, valset):
        super().__init__()
        self.trainset = trainset
        self.valset = valset

    def ask_hyperparameter(self, hp: HyperparameterManager):
        self.image_backbone = hp.suggest_category("image_backbone_stage_2", SUPPORTED_BACKBONES)
        self.batch_size = hp.suggest_int("batch_size", 2, 64, log=True)

    def prepare(self, staff: ModelStaff):
        valloader = DataLoader(self.valset, batch_size=self.batch_size)
        # EvaluationRecipe skips train phase, but framework still expects train/val datasets.
        staff.reg_dataset(self.valset, valloader, self.valset, valloader)

        assert staff.prev_files and len(staff.prev_files) >= 3
        stage2_ckpt = staff.prev_files[-1].get_best_checkpoint_slot()
        logger.info(f"[Validate] loading stage-2 checkpoint from `{stage2_ckpt}`")
        ckpt = torch.load(stage2_ckpt, map_location="cpu")

        clip_model, _ = clip_nopooling.load(self.image_backbone, device="cpu")
        if self.image_backbone == "ViT-L/14@336px":
            utils.enlarge_to_448(clip_model, 336, 448, 14)
        elif self.image_backbone == "RN101":
            utils.enlarge_to_448(clip_model, 224, 448, 32)

        visual_model = AdaptiveClipDecodeStyle(clip_model, self.trainset._num_labels, 0.5)
        estimator = Estimator(torch.zeros((len(self.trainset), self.trainset._num_labels)))
        model = AdaptiveClipWithEstimator(visual_model, estimator)
        model.load_state_dict(ckpt["model"], strict=True)
        staff.reg_model(model)

    def set_optimizers(self, model: AdaptiveClipWithEstimator) -> Optimizer:
        # No optimization is performed in EvaluationRecipe; zero-lr optimizer satisfies the interface.
        return AdamW(model.parameters(), lr=0.0)

    def forward(self, data) -> Any:
        images = torch.cat([data["image"], data["extra_image"]])
        ids = torch.zeros_like(data["index"])
        return self.model(images, ids)

    def cal_loss(self, data, output) -> Tensor:
        bs = data["image"].size(0)
        return F.binary_cross_entropy_with_logits(output[0][:bs], target=data["target"].float())

    def before_epoch(self):
        self.ap = AveragePrecisionMeter(dev=self.dev)

    def after_iter(self, data, output, phase: str):
        bs = data["image"].size(0)
        self.ap.add(output[0][:bs], data["target"], data["name"])

    def report_score(self, phase: str) -> float:
        aps = self.ap.value()
        logger.info(f"[Validate] aps: {aps}")
        return aps.mean().item()


if __name__ == "__main__":
    dataset = chinopie.get_env("dataset")

    tb = TrainBootstrap(
        "deps",
        num_epoch=1,
        load_checkpoint=False,
        save_checkpoint=False,
        comment="interpolation",
        version="1.9.2",
        dataset=dataset,
        enable_prune=False,
        enable_snapshot=False,
        dev="cuda",
        world_size=1,
        seed=9,
    )

    # Keep defaults aligned with main training recipe while allowing CLI override.
    tb.hp.reg_int("batch_size", 16)
    tb.hp.reg_category("image_backbone_stage_2", "RN101")

    trainset, valset = get_dataset(
        tb.file,
        dataset,
        preprocess=transform_zeroshot,
        extra_preprocess=transform_train_hard,
    )

    # Stage=3 reuses stage-2 checkpoint as previous stage output.
    tb.optimize(
        ValidateRecipe(trainset, valset),
        direction="maximize",
        inf_score=0,
        n_trials=1,
        num_epoch=1,
        stage=3,
        always_run=True,
    )
