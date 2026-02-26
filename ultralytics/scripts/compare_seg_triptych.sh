CUDA_VISIBLE_DEVICES=3 \
python /home/r13922151/ultralytics/ultralytics/scripts/compare_seg_triptych.py \
  --data /home/r13922151/ultralytics/ultralytics/cfg/datasets/dsb2018.yaml \
  --split val \
  --baseline-ckpt /home/r13922151/ultralytics/runs/segment/DSB2018/yolov8-seg-baseline4/weights/best.pt \
  --ssm-ckpt /home/r13922151/ultralytics/runs/segment/DSB2018/yolov8-seg-ssm-pan9/weights/best.pt \
  --outdir /home/r13922151/ultralytics/runs/segment/compare_vis/baseline4_vs_ssm-pan9_val_conf025 \
  --imgsz 640 --conf 0.25 --iou 0.7 --max-det 300 --device 0 --retina-masks \
  --alpha 0.35 --line-width 1 --save-summary-csv
