# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.utils.ocd_realization import decode_masks_with_ocd


def _resolve_segment_head(model):
    """Resolve the underlying segment head from wrapped or unwrapped models."""
    core = getattr(model, "model", None)
    if isinstance(core, torch.nn.Sequential):
        return core[-1]
    nested = getattr(core, "model", None)
    if isinstance(nested, torch.nn.Sequential):
        return nested[-1]
    return None


class SegmentationPredictor(DetectionPredictor):
    """A class extending the DetectionPredictor class for prediction based on a segmentation model.

    This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the
    prediction results.

    Attributes:
        args (dict): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO segmentation model.
        batch (list): Current batch of images being processed.

    Methods:
        postprocess: Apply non-max suppression and process segmentation detections.
        construct_results: Construct a list of result objects from predictions.
        construct_result: Construct a single result object from a prediction.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.segment import SegmentationPredictor
        >>> args = dict(model="yolo26n-seg.pt", source=ASSETS)
        >>> predictor = SegmentationPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the SegmentationPredictor with configuration, overrides, and callbacks.

        This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the
        prediction results.

        Args:
            cfg (dict): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (list, optional): List of callback functions to be invoked during prediction.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"

    def postprocess(self, preds, img, orig_imgs):
        """Apply non-max suppression and process segmentation detections for each image in the input batch.

        Args:
            preds (tuple): Model predictions, containing bounding boxes, scores, classes, and mask coefficients.
            img (torch.Tensor): Input image tensor in model format, with shape (B, C, H, W).
            orig_imgs (list | torch.Tensor | np.ndarray): Original image or batch of images.

        Returns:
            (list): List of Results objects containing the segmentation predictions for each image in the batch. Each
                Results object includes both bounding boxes and segmentation masks.

        Examples:
            >>> predictor = SegmentationPredictor(overrides=dict(model="yolo26n-seg.pt"))
            >>> results = predictor.postprocess(preds, img, orig_img)
        """
        self.ocd_extras = preds[1] if isinstance(preds, tuple) and len(preds) > 1 and isinstance(preds[1], dict) else None
        # Extract protos - tuple if PyTorch model or array if exported
        protos = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        return super().postprocess(preds[0], img, orig_imgs, protos=protos)

    def construct_results(self, preds, img, orig_imgs, protos):
        """Construct a list of result objects from the predictions.

        Args:
            preds (list[torch.Tensor]): List of predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_imgs (list[np.ndarray]): List of original images before preprocessing.
            protos (list[torch.Tensor]): List of prototype masks.

        Returns:
            (list[Results]): List of result objects containing the original images, image paths, class names, bounding
                boxes, and masks.
        """
        extras = None
        if isinstance(self.ocd_extras, dict):
            extras = [
                {k: (v[i] if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == len(preds) else v) for k, v in self.ocd_extras.items()}
                for i in range(len(preds))
            ]
        return [
            self.construct_result(pred, img, orig_img, img_path, proto, None if extras is None else extras[i])
            for i, (pred, orig_img, img_path, proto) in enumerate(zip(preds, orig_imgs, self.batch[0], protos))
        ]

    def construct_result(self, pred, img, orig_img, img_path, proto, ocd_extra=None):
        """Construct a single result object from the prediction.

        Args:
            pred (torch.Tensor): The predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_img (np.ndarray): The original image before preprocessing.
            img_path (str): The path to the original image.
            proto (torch.Tensor): The prototype masks.

        Returns:
            (Results): Result object containing the original image, image path, class names, bounding boxes, and masks.
        """
        if pred.shape[0] == 0:  # save empty boxes
            masks = None
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = self._decode_masks(pred, proto, img, orig_img, native=True, ocd_extra=ocd_extra)
        else:
            masks = self._decode_masks(pred, proto, img, orig_img, native=False, ocd_extra=ocd_extra)
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.amax((-2, -1)) > 0  # only keep predictions with masks
            if not all(keep):  # most predictions have masks
                pred, masks = pred[keep], masks[keep]  # indexing is slow
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)

    def _decode_masks(self, pred, proto, img, orig_img, native: bool, ocd_extra=None):
        """Decode masks, optionally using the shared FA-OCD realization bridge."""
        head = _resolve_segment_head(self.model)
        nm = int(getattr(head, "nm", pred.shape[1] - 6))
        apd_enabled = bool(getattr(head, "enable_apd_prior", False))
        coeff = pred[:, 6 : 6 + nm]
        apd_code = pred[:, 6 + nm : 6 + nm + 4] if apd_enabled and pred.shape[1] >= 6 + nm + 4 else None
        if ocd_extra is None or not getattr(head, "ocd_enabled", False):
            if native:
                return ops.process_mask_native(proto, coeff, pred[:, :4], orig_img.shape[:2])
            return ops.process_mask(proto, coeff, pred[:, :4], img.shape[2:], upsample=True)

        image_shape = orig_img.shape[:2] if native else img.shape[2:]
        masks, _ = decode_masks_with_ocd(
            proto=proto,
            mask_coeff=coeff,
            boxes=pred[:, :4],
            shape=image_shape,
            mult_map=ocd_extra.get("ocd_mult"),
            rho_map=ocd_extra.get("ocd_rho"),
            xi_map=ocd_extra.get("ocd_xi"),
            apd_code=apd_code,
            native=native,
            ambiguity_threshold=float(getattr(head, "ambiguity_threshold", 0.35)),
            realization_alpha=float(getattr(head, "realization_alpha", 0.25)),
            apd_gamma=float(getattr(head, "apd_gamma", 0.1)),
            xi_bridge_weight=float(getattr(head, "xi_bridge_weight", 0.15)),
            xi_only=bool(getattr(head, "xi_only", False)),
            stability_threshold=float(getattr(head, "stability_threshold", 0.2)),
            upsample=not native,
        )
        return masks
