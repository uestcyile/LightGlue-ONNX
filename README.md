<div align="right"> English | <a href="https://github.com/fabio-sim/LightGlue-ONNX/blob/main/docs/README.zh.md">简体中文</a></div>

# LightGlue ONNX

Open Neural Network Exchange (ONNX) compatible implementation of [LightGlue: Local Feature Matching at Light Speed](https://github.com/cvg/LightGlue). The ONNX model format allows for interoperability across different platforms with support for multiple execution providers, and removes Python-specific dependencies such as PyTorch. Supports TensorRT and OpenVINO.

<p align="center"><a href="https://arxiv.org/abs/2306.13643"><img src="assets/easy_hard.jpg" alt="LightGlue figure" width=80%></a>

## Updates

- **19 July 2023**: Add support for TensorRT.
- **13 July 2023**: Add support for Flash Attention.
- **11 July 2023**: Add support for mixed precision.
- **4 July 2023**: Add inference time comparisons.
- **1 July 2023**: Add support for extractor `max_num_keypoints`.
- **30 June 2023**: Add support for DISK extractor.
- **28 June 2023**: Add end-to-end SuperPoint+LightGlue export & inference pipeline.

## ONNX Export

Prior to exporting the ONNX models, please install the [requirements](/requirements.txt) of the original LightGlue repository.

To convert the DISK or SuperPoint and LightGlue models to ONNX, run [`export.py`](/export.py). We provide two types of ONNX exports: individual standalone models, and a combined end-to-end pipeline (recommended for convenience) with the `--end2end` flag.

```bash
python export.py \
  --img_size 512 \
  --extractor_type superpoint \
  --extractor_path weights/superpoint.onnx \
  --lightglue_path weights/superpoint_lightglue.onnx \
  --dynamic
```

- Exporting individually can be useful when intermediate outputs can be cached or precomputed. On the other hand, the end-to-end pipeline can be more convenient.
- Although dynamic axes have been specified, it is recommended to export your own ONNX model with the appropriate input image sizes of your use case.
- Use the `--mp` option to export in mixed precision for more speed gains.
- Enable flash attention with the `--flash` option for even faster speeds. ([Flash Attention](https://github.com/HazyResearch/flash-attention) must be installed for export but is not required during inference.)

If you would like to try out inference right away, you can download ONNX models that have already been exported [here](https://github.com/fabio-sim/LightGlue-ONNX/releases).

## ONNX Inference

With ONNX models in hand, one can perform inference on Python using ONNX Runtime (see [requirements-onnx.txt](/requirements-onnx.txt)).

The LightGlue inference pipeline has been encapsulated into a runner class:

```python
from onnx_runner import LightGlueRunner, load_image, rgb_to_grayscale

image0, scales0 = load_image("assets/sacre_coeur1.jpg", resize=512)
image1, scales1 = load_image("assets/sacre_coeur2.jpg", resize=512)
image0 = rgb_to_grayscale(image0)  # only needed for SuperPoint
image1 = rgb_to_grayscale(image1)  # only needed for SuperPoint

# Create ONNXRuntime runner
runner = LightGlueRunner(
    extractor_path="weights/superpoint.onnx",
    lightglue_path="weights/superpoint_lightglue.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    # TensorrtExecutionProvider, OpenVINOExecutionProvider
)

# Run inference
m_kpts0, m_kpts1 = runner.run(image0, image1, scales0, scales1)
```

Note that the output keypoints have already been rescaled back to the original image sizes.

Alternatively, you can also run [`infer.py`](/infer.py).

```bash
python infer.py \
  --img_paths assets/DSC_0410.JPG assets/DSC_0411.JPG \
  --img_size 512 \
  --lightglue_path weights/superpoint_lightglue.onnx \
  --extractor_type superpoint \
  --extractor_path weights/superpoint.onnx \
  --viz
```

## TensorRT Support

TensorRT inference is supported via the TensorRT Execution Provider in ONNXRuntime. Please follow the [official documentation](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) to install TensorRT. The exported ONNX models (whether standalone or end-to-end) must undergo [shape inference](/tools/symbolic_shape_infer.py) for compatibility with TensorRT:

```bash
python tools/symbolic_shape_infer.py \
  --input weights/superpoint.onnx \
  --output weights/superpoint.onnx \
  --auto_merge

python tools/symbolic_shape_infer.py \
  --input weights/superpoint_lightglue.onnx \
  --output weights/superpoint_lightglue.onnx \
  --auto_merge

CUDA_MODULE_LOADING=LAZY && python infer.py \
  --img_paths assets/DSC_0410.JPG assets/DSC_0411.JPG \
  --lightglue_path weights/superpoint_lightglue.onnx \
  --extractor_type superpoint \
  --extractor_path weights/superpoint.onnx \
  --trt \
  --viz
```

The first run will take longer because TensorRT needs to initialise the `.engine` and `.profile` files. Subsequent runs should use the cached files. Note that the ONNX models should not be exported with `--mp` or `--flash`. Only the SuperPoint extractor type is supported.

## Inference Time Comparison

In general, for smaller numbers of keypoints the ONNX version performs similarly to the PyTorch implementation. However, as the number of keypoints increases, the PyTorch CUDA implementation is faster, whereas ONNX is faster overall for CPU inference. See [EVALUATION.md](/evaluation/EVALUATION.md) for technical details.

<p align="center"><a href="https://github.com/fabio-sim/LightGlue-ONNX/blob/main/evaluation/EVALUATION.md"><img src="assets/latency.png" alt="Latency Comparison" width=90%></a>

## Caveats

As the ONNX Runtime has limited support for features like dynamic control flow, certain configurations of the models cannot be exported to ONNX easily. These caveats are outlined below.

### Feature Extraction

- Only batch size `1` is currently supported. This limitation stems from the fact that different images in the same batch can have varying numbers of keypoints, leading to non-uniform (a.k.a. *ragged*) tensors.

### LightGlue Keypoint Matching

- Since dynamic control flow has limited support in ONNX tracing, by extension, early stopping and adaptive point pruning (the `depth_confidence` and `width_confidence` parameters) are also difficult to export to ONNX.
- Note that the end-to-end version, despite its name, still requires the postprocessing (filtering valid matches) function outside the ONNX model since the `scales` variables need to be passed.

Additionally, the outputs of the ONNX models differ slightly from the original PyTorch models (by a small error on the magnitude of `1e-6` to `1e-5` for the scores/descriptors). Although the cause is still unclear, this could be due to differing implementations or modified dtypes.

## Possible Future Work

- **Support for batch size > 1**: Blocked by the fact that different images can have varying numbers of keypoints. Perhaps max-length padding?
- **Support for dynamic control flow**: Investigating *script-mode* ONNX export instead of *trace-mode*.
- **Quantization Support**

## Credits
If you use any ideas from the papers or code in this repo, please consider citing the authors of [LightGlue](https://arxiv.org/abs/2306.13643) and [SuperPoint](https://arxiv.org/abs/1712.07629) and [DISK](https://arxiv.org/abs/2006.13566). Lastly, if the ONNX versions helped you in any way, please also consider starring this repository.

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

```txt
@article{DBLP:journals/corr/abs-2006-13566,
  author       = {Michal J. Tyszkiewicz and
                  Pascal Fua and
                  Eduard Trulls},
  title        = {{DISK:} Learning local features with policy gradient},
  journal      = {CoRR},
  volume       = {abs/2006.13566},
  year         = {2020},
  url          = {https://arxiv.org/abs/2006.13566},
  eprinttype    = {arXiv},
  eprint       = {2006.13566},
  timestamp    = {Wed, 01 Jul 2020 15:21:23 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2006-13566.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
