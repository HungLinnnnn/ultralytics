# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.eval_seg_metric_for_yolo_legacy import (
    compute_imagewise_pq,
    get_dice_1,
    get_fast_aji,
    pq_stats_from_labels,
    remap_label,
    stack_masks_to_label_map,
)
from ultralytics.utils.instance_metrics import compute_pq_components_from_iou, compute_split_merge_from_iou
from ultralytics.utils.metrics import SegmentMetrics, mask_iou, compute_aji, compute_dice


class SegmentationValidator(DetectionValidator):
    """A class extending the DetectionValidator class for validation based on a segmentation model.

    This validator handles the evaluation of segmentation models, processing both bounding box and mask predictions to
    compute metrics such as mAP for both detection and segmentation tasks.

    Attributes:
        plot_masks (list): List to store masks for plotting.
        process (callable): Function to process masks based on save_json and save_txt flags.
        args (namespace): Arguments for the validator.
        metrics (SegmentMetrics): Metrics calculator for segmentation tasks.
        stats (dict): Dictionary to store statistics during validation.

    Examples:
        >>> from ultralytics.models.yolo.segment import SegmentationValidator
        >>> args = dict(model="yolo26n-seg.pt", data="coco8-seg.yaml")
        >>> validator = SegmentationValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): DataLoader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (namespace, optional): Arguments for the validator.
            _callbacks (list, optional): List of callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.process = None
        self.args.task = "segment"
        self.metrics = SegmentMetrics()
        self.seg_metric_backend = "native"
        self.seg_metric_legacy_pq_reduce = "imagewise"

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch of images for YOLO segmentation validation.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.

        Returns:
            (dict[str, Any]): Preprocessed batch.
        """
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].float()
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics and select mask processing function based on save_json flag.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        super().init_metrics(model)
        if self.args.save_json:
            check_requirements("faster-coco-eval>=1.6.7")
        # More accurate vs faster
        self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask
        self.seg_metric_backend = str(getattr(self.args, "seg_metric_backend", "native")).lower()
        self.seg_metric_legacy_pq_reduce = str(getattr(self.args, "seg_metric_legacy_pq_reduce", "imagewise")).lower()
        if self.seg_metric_backend not in {"native", "legacy"}:
            raise ValueError(
                f"Invalid seg_metric_backend={self.seg_metric_backend!r}. Expected one of: native, legacy"
            )
        if self.seg_metric_backend == "legacy" and self.seg_metric_legacy_pq_reduce != "imagewise":
            raise ValueError(
                "Invalid seg_metric_legacy_pq_reduce="
                f"{self.seg_metric_legacy_pq_reduce!r}. Currently only 'imagewise' is supported."
            )

    def get_desc(self) -> str:
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 17) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "PQ",
            "SQ",
            "RQ",
            "AJI",
            "Dice",
            "SplitR",
            "MergeR",
        )

    def postprocess(self, preds: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """Post-process YOLO predictions and return output detections with proto.

        Args:
            preds (list[torch.Tensor]): Raw predictions from the model.

        Returns:
            list[dict[str, torch.Tensor]]: Processed detection predictions with masks.
        """
        proto = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        preds = super().postprocess(preds[0])
        imgsz = [4 * x for x in proto.shape[2:]]  # get image size from proto
        for i, pred in enumerate(preds):
            coefficient = pred.pop("extra")
            pred["masks"] = (
                self.process(proto[i], coefficient, pred["bboxes"], shape=imgsz)
                if coefficient.shape[0]
                else torch.zeros(
                    (0, *(imgsz if self.process is ops.process_mask_native else proto.shape[2:])),
                    dtype=torch.uint8,
                    device=pred["bboxes"].device,
                )
            )
        return preds

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare a batch for training or inference by processing images and targets.

        Args:
            si (int): Batch index.
            batch (dict[str, Any]): Batch data containing images and annotations.

        Returns:
            (dict[str, Any]): Prepared batch with processed annotations.
        """
        prepared_batch = super()._prepare_batch(si, batch)
        nl = prepared_batch["cls"].shape[0]
        if self.args.overlap_mask:
            masks = batch["masks"][si]
            index = torch.arange(1, nl + 1, device=masks.device).view(nl, 1, 1)
            masks = (masks == index).float()
        else:
            masks = batch["masks"][batch["batch_idx"] == si]
        if nl:
            mask_size = [s if self.process is ops.process_mask_native else s // 4 for s in prepared_batch["imgsz"]]
            if masks.shape[1:] != mask_size:
                masks = F.interpolate(masks[None], mask_size, mode="bilinear", align_corners=False)[0]
                masks = masks.gt_(0.5)
        prepared_batch["masks"] = masks
        return prepared_batch

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """Compute correct prediction matrix for a batch based on bounding boxes and optional masks.

        Args:
            preds (dict[str, torch.Tensor]): Dictionary containing predictions with keys like 'cls' and 'masks'.
            batch (dict[str, Any]): Dictionary containing batch data with keys like 'cls' and 'masks'.

        Returns:
            (dict[str, np.ndarray]): A dictionary containing correct prediction matrices including 'tp_m' for mask IoU.

        Examples:
            >>> preds = {"cls": torch.tensor([1, 0]), "masks": torch.rand(2, 640, 640), "bboxes": torch.rand(2, 4)}
            >>> batch = {"cls": torch.tensor([1, 0]), "masks": torch.rand(2, 640, 640), "bboxes": torch.rand(2, 4)}
            >>> correct_preds = validator._process_batch(preds, batch)

        Notes:
            - If `masks` is True, the function computes IoU between predicted and ground truth masks.
            - If `overlap` is True and `masks` is True, overlapping masks are taken into account when computing IoU.
        """
        tp = super()._process_batch(preds, batch)
        gt_stack = batch["masks"].bool().cpu().numpy()
        pred_stack = preds["masks"].float().gt(0.5).cpu().numpy()
        ng, npred = int(gt_stack.shape[0]), int(pred_stack.shape[0])

        if ng > 0 and npred > 0:
            iou_t = mask_iou(batch["masks"].flatten(1), preds["masks"].flatten(1).float())  # float, uint8
            iou_np = iou_t.cpu().numpy()
            tp_m = self.match_predictions(preds["cls"], batch["cls"], iou_t).cpu().numpy()
        else:
            iou_np = np.zeros((ng, npred), dtype=np.float64)
            tp_m = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)

        pq_comp = compute_pq_components_from_iou(iou_np, thr=0.5, method="auto")
        sm_comp = compute_split_merge_from_iou(iou_np, ng=ng, npred=npred, thr=0.5)
        pq = np.array([pq_comp["pq"]], dtype=float)
        sq = np.array([pq_comp["sq"]], dtype=float)
        rq = np.array([pq_comp["rq"]], dtype=float)
        tp_cnt = np.array([pq_comp["tp"]], dtype=float)
        fp_cnt = np.array([pq_comp["fp"]], dtype=float)
        fn_cnt = np.array([pq_comp["fn"]], dtype=float)
        iou_sum_tp = np.array([pq_comp["iou_sum"]], dtype=float)
        ng_inst = np.array([pq_comp["ng"]], dtype=float)
        np_inst = np.array([pq_comp["npred"]], dtype=float)
        split_count = np.array([sm_comp["split_count"]], dtype=float)
        merge_count = np.array([sm_comp["merge_count"]], dtype=float)
        split_rate = np.array([sm_comp["split_rate"]], dtype=float)
        merge_rate = np.array([sm_comp["merge_rate"]], dtype=float)
        mean_iou_tp = np.array([pq_comp["mean_iou_tp"]], dtype=float)

        gt_cls = batch["cls"]
        if gt_cls.shape[0] == 0 or preds["cls"].shape[0] == 0:
            if self.seg_metric_backend == "legacy" and gt_cls.shape[0] == 0 and preds["cls"].shape[0] == 0:
                aji = np.array([1.0], dtype=float)
                dice = np.array([1.0], dtype=float)
            else:
                aji = np.array([0.0], dtype=float)
                dice = np.array([0.0], dtype=float)
        else:
            if self.seg_metric_backend == "legacy":
                gt_label = remap_label(stack_masks_to_label_map(gt_stack))
                pred_label = remap_label(stack_masks_to_label_map(pred_stack))

                dice = np.array([get_dice_1(gt_label, pred_label)], dtype=float)
                aji = np.array([get_fast_aji(gt_label, pred_label)], dtype=float)
                tp_cnt_legacy, fp_cnt_legacy, fn_cnt_legacy, iou_sum_legacy = pq_stats_from_labels(
                    gt_label, pred_label, match_iou=0.5
                )
                if self.seg_metric_legacy_pq_reduce == "imagewise":
                    _ = compute_imagewise_pq(
                        tp_cnt_legacy, fp_cnt_legacy, fn_cnt_legacy, iou_sum_legacy
                    )  # keep legacy code path behavior
                else:
                    raise ValueError(
                        "Unsupported seg_metric_legacy_pq_reduce="
                        f"{self.seg_metric_legacy_pq_reduce!r}."
                    )
            else:
                # PQ/AJI/Dice
                gt_masks = batch["masks"].flatten(1).bool()
                pred_masks = preds["masks"].flatten(1).float().gt(0.5)
                matches = np.argwhere(iou_np >= 0.5)
                if matches.shape[0]:
                    matches = matches[np.argsort(iou_np[matches[:, 0], matches[:, 1]])[::-1]]
                    used_g, used_p, kept = set(), set(), []
                    for g, p in matches:
                        if g not in used_g and p not in used_p:
                            used_g.add(int(g))
                            used_p.add(int(p))
                            kept.append((int(g), int(p)))
                else:
                    kept = []

                gt_np = gt_masks.cpu().numpy()
                pred_np = pred_masks.cpu().numpy()
                aji = np.array([compute_aji(gt_np, pred_np, kept)], dtype=float)

                gt_union = gt_np.any(axis=0)
                pred_union = pred_np.any(axis=0)
                dice = np.array([compute_dice(gt_union, pred_union)], dtype=float)
        tp.update(
            {
                "tp_m": tp_m,
                "pq": pq,
                "sq": sq,
                "rq": rq,
                "aji": aji,
                "dice": dice,
                "tp_cnt": tp_cnt,
                "fp_cnt": fp_cnt,
                "fn_cnt": fn_cnt,
                "iou_sum_tp": iou_sum_tp,
                "ng_inst": ng_inst,
                "np_inst": np_inst,
                "split_count": split_count,
                "merge_count": merge_count,
                "split_rate": split_rate,
                "merge_rate": merge_rate,
                "mean_iou_tp": mean_iou_tp,
            }
        )  # update tp with mask IoU and extra metrics
        return tp

    def plot_predictions(self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int) -> None:
        """Plot batch predictions with masks and bounding boxes.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.
            preds (list[dict[str, torch.Tensor]]): List of predictions from the model.
            ni (int): Batch index.
        """
        for p in preds:
            masks = p["masks"]
            if masks.shape[0] > self.args.max_det:
                LOGGER.warning(f"Limiting validation plots to 'max_det={self.args.max_det}' items.")
            p["masks"] = torch.as_tensor(masks[: self.args.max_det], dtype=torch.uint8).cpu()
        super().plot_predictions(batch, preds, ni, max_det=self.args.max_det)  # plot bboxes

    def save_one_txt(self, predn: torch.Tensor, save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """Save YOLO detections to a txt file in normalized coordinates in a specific format.

        Args:
            predn (torch.Tensor): Predictions in the format (x1, y1, x2, y2, conf, class).
            save_conf (bool): Whether to save confidence scores.
            shape (tuple[int, int]): Shape of the original image.
            file (Path): File path to save the detections.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            masks=torch.as_tensor(predn["masks"], dtype=torch.uint8),
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Save one JSON result for COCO evaluation.

        Args:
            predn (dict[str, torch.Tensor]): Predictions containing bboxes, masks, confidence scores, and classes.
            pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.
        """

        def to_string(counts: list[int]) -> str:
            """Converts the RLE object into a compact string representation. Each count is delta-encoded and
            variable-length encoded as a string.

            Args:
                counts (list[int]): List of RLE counts.
            """
            result = []

            for i in range(len(counts)):
                x = int(counts[i])

                # Apply delta encoding for all counts after the second entry
                if i > 2:
                    x -= int(counts[i - 2])

                # Variable-length encode the value
                while True:
                    c = x & 0x1F  # Take 5 bits
                    x >>= 5

                    # If the sign bit (0x10) is set, continue if x != -1;
                    # otherwise, continue if x != 0
                    more = (x != -1) if (c & 0x10) else (x != 0)
                    if more:
                        c |= 0x20  # Set continuation bit
                    c += 48  # Shift to ASCII
                    result.append(chr(c))
                    if not more:
                        break

            return "".join(result)

        def multi_encode(pixels: torch.Tensor) -> list[int]:
            """Convert multiple binary masks using Run-Length Encoding (RLE).

            Args:
                pixels (torch.Tensor): A 2D tensor where each row represents a flattened binary mask with shape [N,
                    H*W].

            Returns:
                (list[int]): A list of RLE counts for each mask.
            """
            transitions = pixels[:, 1:] != pixels[:, :-1]
            row_idx, col_idx = torch.where(transitions)
            col_idx = col_idx + 1

            # Compute run lengths
            counts = []
            for i in range(pixels.shape[0]):
                positions = col_idx[row_idx == i]
                if len(positions):
                    count = torch.diff(positions).tolist()
                    count.insert(0, positions[0].item())
                    count.append(len(pixels[i]) - positions[-1].item())
                else:
                    count = [len(pixels[i])]

                # Ensure starting with background (0) count
                if pixels[i][0].item() == 1:
                    count = [0, *count]
                counts.append(count)

            return counts

        pred_masks = predn["masks"].transpose(2, 1).contiguous().view(len(predn["masks"]), -1)  # N, H*W
        h, w = predn["masks"].shape[1:3]
        counts = multi_encode(pred_masks)
        rles = []
        for c in counts:
            rles.append({"size": [h, w], "counts": to_string(c)})
        super().pred_to_json(predn, pbatch)
        for i, r in enumerate(rles):
            self.jdict[-len(rles) + i]["segmentation"] = r  # segmentation

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Scales predictions to the original image size."""
        return {
            **super().scale_preds(predn, pbatch),
            "masks": ops.scale_masks(predn["masks"][None], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])[
                0
            ].byte(),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Return COCO-style instance segmentation evaluation metrics."""
        pred_json = self.save_dir / "predictions.json"  # predictions
        anno_json = (
            self.data["path"]
            / "annotations"
            / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        )  # annotations
        return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "segm"], suffix=["Box", "Mask"])
