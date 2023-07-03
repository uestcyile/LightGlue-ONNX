# Evaluation

The inference time of LightGlue-ONNX is compared to that of the original PyTorch implementation with default configuration.

## Methods

Following the implementation details of the [LightGlue paper](https://arxiv.org/abs/2306.13643), we report the inference time, or latency, of only the LightGlue matcher; that is, the time taken for feature extraction, postprocessing, copying data between the host & device, or finding inliers (e.g., CONSAC/MAGSAC) is not measured. The average inference time is defined as the mean over all samples in the [MegaDepth](https://arxiv.org/abs/1804.00607) test dataset. We use the data provided by [LoFTR](https://arxiv.org/abs/2104.00680) [here](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md) - a total of 403 image pairs.

Each image is resized such that its longer side is 1024 before being fed into the feature extractor. The average inference time of the LightGlue matcher is then measured for different values of the extractor's `max_num_keypoints` parameter: 512, 1024, 2048, and 4096. The [SuperPoint](http://arxiv.org/abs/1712.07629) extractor is used.

All experiments are conducted on a [Google Colab](https://colab.research.google.com/github/fabio-sim/LightGlue-ONNX/blob/main/evaluation/lightglue-onnx.ipynb) GPU Runtime (Tesla T4).

## Results

The measured run times are plotted in the figure below.

![Latency vs. Number of Keypoints](../assets/latency.png)

<table align="center"><thead><tr><th>Number of Keypoints</th><th></th><th>512</th><th>1024</th><th>2048</th><th>4096</th></tr><tr><th>Model</th><th>Device</th><th colspan="4">Latency (ms)</th></tr></thead><tbody><tr><td>LightGlue</td><td>CUDA</td><td>35.42</td><td>47.36</td><td>112.87</td><td>187.51</td></tr><tr><td>LightGlue-ONNX</td><td>CUDA</td><td>30.44</td><td>82.24</td><td>269.39</td><td>519.41</td></tr><tr><td>LightGlue</td><td>CPU</td><td>1121</td><td>3818</td><td>15968</td><td>37587</td></tr><tr><td>LightGlue-ONNX</td><td>CPU</td><td>759</td><td>2961</td><td>10493</td><td>20143</td></tr></tbody></table>

At smaller numbers of keypoints, the difference between the CUDA ONNX and PyTorch latencies are small; however, this becomes much more noticeable at higher keypoint numbers, where PyTorch is faster. The cause remains to be investigated (different operator implementations?). On the other hand, ONNX is faster overall for CPU inference.
