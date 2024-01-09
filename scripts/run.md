## run infer with onnx
python infer.py   --img_paths assets/1699355803784416512.jpg assets/1699355805160020992.jpg   --img_size 640   --lightglue_path weights/superpoint_lightglue.onnx   --extractor_type superpoint   --extractor_path /home/data/zhyile/code/sw_hdmap_cloud/src/hd_map/algorithms/super_features/models/superpoint_48dim_512_repvggv_113_simple.onnx   --viz

## trans onnx to tensorrt
# shape inference
python tools/symbolic_shape_infer.py --input weights/superpoint_lightglue.onnx --output weights/superpoint_lightglue_shape_infered.onnx
# to tensorrt
/home/data/zhyile/packages/TensorRT-8.2.1.8.Linux.x86_64-gnu.cuda-11.4.cudnn8.2/TensorRT-8.2.1.8/bin/trtexec --onnx=/home/data/zhyile/code/LightGlue-ONNX/weights/superpoint_lightglue_shape_infered.onnx --saveEngine=/home/data/zhyile/code/LightGlue-ONNX/weights/superpoint_lightglue_shape_infered_fp16.trt  --minShapes=kpts0:1x300x2,kpts1:1x300x2,desc0:1x300x48,desc1:1x300x48 --optShapes=kpts0:1x512x2,kpts1:1x512x2,desc0:1x512x48,desc1:1x512x48 --maxShapes=kpts0:1x512x2,kpts1:1x512x2,desc0:1x512x48,desc1:1x512x48 --explicitBatch --workspace=4096 --fp16 --verbose
