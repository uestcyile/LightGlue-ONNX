# LightGlue ONNX

Open Neural Network Exchange (ONNX) compatible implementation of [LightGlue: Local Feature Matching at Light Speed](https://github.com/cvg/LightGlue). The ONNX model format allows for interoperability across different platforms with support for multiple execution providers, and removes Python-specific dependencies such as PyTorch.

<p align="center"><a href="https://arxiv.org/abs/2306.13643"><img src="assets/easy_hard.jpg" alt="LightGlue figure" width=80%></a>

## Updates

- **28 June 2023**: Add end-to-end SuperPoint+LightGlue export & inference pipeline.
## ONNX Export

Prior to exporting the ONNX models, please install the [requirements](./requirements.txt) of the original LightGlue repository. ([Flash Attention](https://github.com/HazyResearch/flash-attention) does not need to be installed.)

We provide two types of ONNX exports of the SuperPoint+LightGlue pipeline: individual models, and a combined end-to-end pipeline (recommended for convenience).

- To convert the SuperPoint and LightGlue models to ONNX, run the following script:

```bash
python export_superpoint_lightglue.py --img_size 512 --superpoint_path weights/superpoint.onnx --lightglue_path weights/superpoint_lightglue.onnx
```

Exporting individually can be useful when intermediate outputs can be cached or precomputed. On the other hand, the end-to-end pipeline can be more convenient.

- To combine both into a single ONNX model, run the following script:

```bash
python export_superpoint_lightglue_end2end.py --img_size 512 --save_path weights/superpoint_lightglue_end2end.onnx
```

Although dynamic axes have been specified, it is recommended to export your own ONNX model with the appropriate input image sizes of your use case.

## ONNX Inference

With ONNX models in hand, one can perform inference on Python using ONNX Runtime (see [requirements-onnx.txt](./requirements-onnx.txt)).

The SuperPoint+LightGlue inference pipeline has been encapsulated into a runner class:

```python
from onnx_runner import (
    SuperPointLightGlueEnd2EndRunner,
    SuperpointLightglueRunner,
    load_image,
    rgb_to_grayscale,
)

img0_path = "assets/sacre_coeur1.jpg"
img1_path = "assets/sacre_coeur2.jpg"
size = 512
image0, scales0 = load_image(img0_path, resize=size)
image1, scales1 = load_image(img1_path, resize=size)
image0 = rgb_to_grayscale(image0)
image1 = rgb_to_grayscale(image1)

# Create ONNXRuntime runner
# Separate ONNX models (export_superpoint_lightglue.py)
runner = SuperpointLightglueRunner(
    superpoint_path="weights/superpoint.onnx",
    lightglue_path="weights/superpoint_lightglue.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

# Combined model (export_superpoint_lightglue_end2end.py)
runner = SuperPointLightGlueEnd2EndRunner(
    onnx_path="weights/superpoint_lightglue_end2end.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

# Run inference
m_kpts0, m_kpts1 = runner.run(image0, image1, scales0, scales1)

```

## Caveats

As the ONNX Runtime has limited support for features like dynamic control flow, certain configurations of the models cannot be exported to ONNX easily. These caveats are outlined below.

### Feature Extraction

- The `DISK` extractor cannot be exported to ONNX due to the use of `same` padding in the convolution layers of its UNet.
- The `max_num_keypoints` parameter (i.e., setting an upper bound on the number of keypoints returned by the extractor) is not supported at the moment due to `torch.topk()`.
- Grayscale images are assumed. The conditional RGB2Grayscale logic that was previously included in SuperPoint's forward pass has been moved outside and refactored into the `rgb_to_grayscale()` utility function.
- Only batch size `1` is currently supported. This limitation stems from the fact that different images in the same batch can have varying numbers of keypoints, leading to non-uniform (a.k.a. *ragged*) tensors.

### LightGlue Keypoint Matching

- Since dynamic control flow has limited support in ONNX tracing, by extension, early stopping and adaptive point pruning (the `depth_confidence` and `width_confidence` parameters) are also difficult to export to ONNX.
- PyTorch's `F.scaled_dot_product_attention()` function fails to export to ONNX as of version `2.0.1`. However, this [issue](https://github.com/pytorch/pytorch/issues/97262) seems to be fixed in PyTorch-nightly. Currently, the backup implementation (`else` branch of `lightglue_onnx.lightglue.FastAttention.forward`) is used.
- Mixed precision is turned off.
- Note that the end-to-end version, despite its name, still requires the postprocessing (filtering valid matches) function outside the ONNX model since the `scales` variables need to be passed.

Additionally, the outputs of the ONNX models differ slightly from the original PyTorch models (by a small error on the magnitude of `1e-6` to `1e-5` for the scores/descriptors). Although the cause is still unclear, this could be due to differing implementations or modified dtypes.

## Possible Future Work

- **Support for TensorRT**: Appears to be currently blocked by unsupported Einstein summation operations (`torch.einsum()`) in TensorRT - Thanks to [Shidqiet](https://github.com/Shidqiet)'s investigation.
- **Support for batch size > 1**: Blocked by the fact that different images can have varying numbers of keypoints. Perhaps max-length padding?
- **Support for dynamic control flow**: Investigating *script-mode* ONNX export instead of *trace-mode*.
- **Mixed-precision Support**

## Credits
If you use any ideas from the papers or code in this repo, please consider citing the authors of [LightGlue](https://arxiv.org/abs/2306.13643) and [SuperPoint](https://arxiv.org/abs/1712.07629):

```txt
@inproceedings{lindenberger23lightglue,
  author    = {Philipp Lindenberger and
               Paul-Edouard Sarlin and
               Marc Pollefeys},
  title     = {{LightGlue}: Local Feature Matching at Light Speed},
  booktitle = {ArXiv PrePrint},
  year      = {2023}
}
```

```txt
@article{DBLP:journals/corr/abs-1712-07629,
  author       = {Daniel DeTone and
                  Tomasz Malisiewicz and
                  Andrew Rabinovich},
  title        = {SuperPoint: Self-Supervised Interest Point Detection and Description},
  journal      = {CoRR},
  volume       = {abs/1712.07629},
  year         = {2017},
  url          = {http://arxiv.org/abs/1712.07629},
  eprinttype    = {arXiv},
  eprint       = {1712.07629},
  timestamp    = {Mon, 13 Aug 2018 16:47:29 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1712-07629.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
