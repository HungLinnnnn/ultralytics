cd /home/r13922151/ultralytics/ultralytics
GPU=7
CUDA_VISIBLE_DEVICES=$GPU \
/home/r13922151/miniconda3/envs/yolo-ssm/bin/python scripts/visualize_seg_prototypes.py \
  --weights /home/r13922151/ultralytics/yolov8n-seg.pt \
  --source /home/r13922151/cell_datasets/dataset/dsb2018/stardist_yolo/test/images/0bda515e370294ed94efd36bd53782288acacb040c171df2ed97fd691fc9d8fe.tif \
  --out-dir /home/r13922151/ultralytics/runs/segment/prototype_vis_smoke_pkg \
  --imgsz 640 \
  --device $GPU \
  --conf 0.25 \
