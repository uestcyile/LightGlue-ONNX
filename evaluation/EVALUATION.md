# Evaluation

The inference time of LightGlue-ONNX is compared to that of the original PyTorch implementation with default configuration.

## Methods

Following the implementation details of the [LightGlue paper](https://arxiv.org/abs/2306.13643), we report the inference time, or latency, of only the LightGlue matcher; that is, the time taken for feature extraction, postprocessing, copying data between the host & device, or finding inliers (e.g., CONSAC/MAGSAC) is not measured. The average inference time is defined as the mean over all samples in the [MegaDepth](https://arxiv.org/abs/1804.00607) test dataset. We use the data provided by [LoFTR](https://arxiv.org/abs/2104.00680) [here](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md) - a total of 403 image pairs.

Each image is resized such that its longer side is 1024 before being fed into the feature extractor. The average inference time of the LightGlue matcher is then measured for different values of the extractor's `max_num_keypoints` parameter: 512, 1024, 2048, and 4096. The [SuperPoint](http://arxiv.org/abs/1712.07629) extractor is used.

All experiments are conducted on a [Google Colab](lightglue-onnx.ipynb) GPU Runtime (Tesla T4) with `CUDA==11.8.1` and `TensorRT==8.6.1`.

## Results

The measured run times are plotted in the figure below.

<p align="center"><a href="https://github.com/fabio-sim/LightGlue-ONNX/blob/main/evaluation/EVALUATION.md"><img src="../assets/latency.png" alt="Latency Comparison" width=100%></a>

<table align="center"><thead><tr><th>Number of Keypoints</th><th></th><th>512</th><th>1024</th><th>2048</th><th>4096</th></tr><tr><th>Model</th><th>Device</th><th colspan="4">Latency (ms)</th></tr></thead><tbody><tr><td>LightGlue</td><td>CUDA</td><td>35.42</td><td>47.36</td><td>112.87</td><td>187.51</td></tr><tr><td>LightGlue-ONNX</td><td>CUDA</td><td>30.44</td><td>82.24</td><td>269.39</td><td>519.41</td></tr><tr><td>LightGlue-MP</td><td>CUDA</td><td>36.32</td><td>37.10</td><td>61.58</td><td>127.59</td></tr><tr><td>LightGlue-ONNX-MP</td><td>CUDA</td><td>24.2</td><td>66.27</td><td>227.91</td><td>473.71</td></tr><tr><td>LightGlue-MP-Flash</td><td>CUDA</td><td>38.3</td><td>38.8</td><td>42.9</td><td>55.9</td></tr><tr><td>LightGlue-ONNX-MP-Flash</td><td>CUDA</td><td>21.2</td><td>57.4</td><td>191.1</td><td>368.9</td></tr><tr><td>LightGlue-ONNX-TRT</td><td>TensorRT-CUDA</td><td>7.08</td><td>15.88</td><td>47.04</td><td>107.89</td></tr><tr><td>LightGlue</td><td>CPU</td><td>1121</td><td>3818</td><td>15968</td><td>37587</td></tr><tr><td>LightGlue-ONNX</td><td>CPU</td><td>759</td><td>2961</td><td>10493</td><td>20143</td></tr></tbody></table>

At smaller numbers of keypoints, the difference between the CUDA ONNX and PyTorch latencies are small; however, this becomes much more noticeable at higher keypoint numbers, where PyTorch is faster. The cause remains to be investigated (different operator implementations?). On the other hand, ONNX is faster overall for CPU inference.

## TensorRT

Note that TensorRT incurs an upfront initialisation cost in order to build the `.engine` and `.profile` files during the first run (subsequent runs can use the cached files). Depending on the machine, this build time can take more than 10 minutes to complete. When using dynamic axes with the TensorRT Execution Provider, it is recommended to pass the min-opt-max shape range options in order to prevent TensorRT from having to rebuild a new runtime profile whenever an unexpected shape is encountered. Corresponding snippet from [`eval.py`](/eval.py):

```python
import onnxruntime as ort

max_num_keypoints = 512  # 1024, 2048

trt_ep_options = {
    "trt_fp16_enable": True,
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": "weights/cache",
    "trt_profile_min_shapes": f"kpts0:1x1x2,kpts1:1x1x2,desc0:1x1x256,desc1:1x1x256",
    "trt_profile_opt_shapes": f"kpts0:1x{max_num_keypoints}x2,kpts1:1x{max_num_keypoints}x2,desc0:1x{max_num_keypoints}x256,desc1:1x{max_num_keypoints}x256",
    "trt_profile_max_shapes": f"kpts0:1x{max_num_keypoints}x2,kpts1:1x{max_num_keypoints}x2,desc0:1x{max_num_keypoints}x256,desc1:1x{max_num_keypoints}x256"
}

sess = ort.InferenceSession(
    "superpoint_lightglue.onnx",
    providers=[
        ("TensorrtExecutionProvider", trt_ep_options),
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
)
```
