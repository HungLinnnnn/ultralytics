# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


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
        # Extract protos - tuple if PyTorch model or array if exported
        protos = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        return super().postprocess(preds[0], img, orig_imgs, protos=protos)

    def construct_results(self, preds, img, orig_imgs, protos, h4_export=None):
        """Construct a list of result objects from the predictions.

        Args:
            preds (list[torch.Tensor]): List of predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_imgs (list[np.ndarray]): List of original images before preprocessing.
            protos (list[torch.Tensor]): List of prototype masks.
            h4_export (dict, optional): H4 internals metadata captured around NMS.

        Returns:
            (list[Results]): List of result objects containing the original images, image paths, class names, bounding
                boxes, and masks.
        """
        h4_images = [None] * len(preds)
        if h4_export:
            h4_images = []
            for i in range(len(preds)):
                h4_images.append(
                    {
                        "pre_nms_candidates": h4_export["pre_nms_candidates"][i],
                        "kept_pre_nms_indices": h4_export["kept_pre_nms_indices"][i],
                        "nms_config": h4_export["nms_config"],
                        "input_tensor_shape": h4_export["input_tensor_shape"],
                    }
                )
        return [
            self.construct_result(pred, img, orig_img, img_path, proto, h4_image)
            for pred, orig_img, img_path, proto, h4_image in zip(preds, orig_imgs, self.batch[0], protos, h4_images)
        ]

    def construct_result(self, pred, img, orig_img, img_path, proto, h4_export=None):
        """Construct a single result object from the prediction.

        Args:
            pred (torch.Tensor): The predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_img (np.ndarray): The original image before preprocessing.
            img_path (str): The path to the original image.
            proto (torch.Tensor): The prototype masks.
            h4_export (dict, optional): Per-image H4 internals metadata captured around NMS.

        Returns:
            (Results): Result object containing the original image, image path, class names, bounding boxes, and masks.
        """
        h4_enabled = h4_export is not None
        h4_pred_pre_mask_filter = pred.clone() if h4_enabled else None
        h4_mask_logits = None
        h4_mask_keep = None
        h4_mask_process = "empty"
        if pred.shape[0] == 0:  # save empty boxes
            masks = None
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            if h4_enabled and getattr(self.args, "h4_export_logits", False):
                from ultralytics.utils.h4_export import compute_mask_logits

                h4_mask_logits = compute_mask_logits(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2], True)
            masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # NHW
            h4_mask_process = "process_mask_native"
        else:
            if h4_enabled and getattr(self.args, "h4_export_logits", False):
                from ultralytics.utils.h4_export import compute_mask_logits

                h4_mask_logits = compute_mask_logits(proto, pred[:, 6:], pred[:, :4], img.shape[2:], False)
            masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # NHW
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            h4_mask_process = "process_mask"
        if masks is not None:
            keep = masks.amax((-2, -1)) > 0  # only keep predictions with masks
            h4_mask_keep = keep.clone() if h4_enabled else None
            if not all(keep):  # most predictions have masks
                pred, masks = pred[keep], masks[keep]  # indexing is slow
                if h4_mask_logits is not None:
                    h4_mask_logits = h4_mask_logits[keep]
        elif h4_enabled:
            h4_mask_keep = pred.new_zeros((0,)).bool()

        if h4_enabled:
            from ultralytics.utils.h4_export import write_segmentation_internals

            kept_pre_nms = h4_export["kept_pre_nms_indices"]
            kept_after_mask = (
                kept_pre_nms[h4_mask_keep] if h4_mask_keep is not None and kept_pre_nms.numel() else kept_pre_nms
            )
            pre_nms_candidates = h4_export["pre_nms_candidates"]
            metadata = {
                "dataset": getattr(self.args, "data", None),
                "split": getattr(self.args, "split", None),
                "checkpoint_path": getattr(self.args, "model", None),
                "imgsz": getattr(self.args, "imgsz", None),
                "retina_masks": self.args.retina_masks,
                "orig_shape": list(orig_img.shape[:2]),
                "input_tensor_shape": h4_export["input_tensor_shape"],
                "proto_shape": list(proto.shape),
                "mask_process": h4_mask_process,
                "mask_logits_saved": h4_mask_logits is not None,
                "pre_mask_filter_count": int(h4_pred_pre_mask_filter.shape[0]),
                "post_mask_filter_count": int(pred.shape[0]),
                "nms_config": h4_export["nms_config"],
            }
            arrays = {
                "proto": proto,
                "pre_nms_candidate_indices": pre_nms_candidates["indices"],
                "pre_nms_candidate_rows": pre_nms_candidates["rows"],
                "kept_pre_nms_indices_pre_mask_filter": kept_pre_nms,
                "kept_pre_nms_indices": kept_after_mask,
                "post_nms_rows_pre_mask_filter": h4_pred_pre_mask_filter,
                "post_nms_mask_keep": h4_mask_keep,
                "post_nms_rows": pred,
                "boxes_conf_cls_post_nms": pred[:, :6],
                "coeff_post_nms": pred[:, 6:],
                "masks_post_nms": masks,
                "mask_logits_post_nms": h4_mask_logits,
                "prediction_id_post_nms": kept_pre_nms.new_tensor(range(pred.shape[0])),
            }
            write_segmentation_internals(
                self.save_dir,
                getattr(self.args, "h4_export_dir", None),
                img_path,
                metadata,
                arrays,
            )
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)
