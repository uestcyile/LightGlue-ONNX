<div align="right"> <a href="https://github.com/fabio-sim/LightGlue-ONNX">English</a> | 简体中文</div>

# LightGlue ONNX

支持Open Neural Network Exchange (ONNX)的[LightGlue: Local Feature Matching at Light Speed](https://github.com/cvg/LightGlue)实施。ONNX格式支持不同平台之间的互操作性，并支持多个执行提供程序，同时消除了Python特定的依赖项，比如PyTorch。支持TensorRT和OpenVINO。

<p align="center"><a href="https://arxiv.org/abs/2306.13643"><img src="../assets/easy_hard.jpg" alt="LightGlue figure" width=80%></a>

## 更新

- **2023年7月19日**: 支持TensorRT。
- **2023年7月13日**: 支持FlashAttention。
- **2023年7月11日**: 支持混合精度。
- **2023年7月4日**: 加了运行时间比较。
- **2023年7月1日**: 支持特征提取`max_num_keypoints`。
- **2023年6月30日**: 支持DISK特征提取。
- **2023年6月28日**: 加了端到端SuperPoint+LightGlue转换。

## ONNX格式转换

在转换ONNX模型之前，请安装原始LightGlue的[requirements](/requirements.txt)。

将DISK或SuperPoint和LightGlue模型转换为ONNX格式，请运行[`export.py`](/export.py)。提供了两种类型的ONNX转换：独立模型和组合模型(使用`--end2end`，比较方便)。

```bash
python export.py \
  --img_size 512 \
  --extractor_type superpoint \
  --extractor_path weights/superpoint.onnx \
  --lightglue_path weights/superpoint_lightglue.onnx \
  --dynamic
```

- 虽然已指定了`--dynamic`，但建议使用适合您用例的图像大小转换。
- 指定`--mp`使混合精度。
- 指定`--flash`使FlashAttention。(ONNX格式转换需要安装[Flash Attention](https://github.com/HazyResearch/flash-attention)，但推理不需要。)

如果您想立即尝试ONNX运行，可以下载已转换的[ONNX模型](https://github.com/fabio-sim/LightGlue-ONNX/releases)。

## ONNX推理

有了ONNX模型，就可以使用ONNX Runtime进行推理(请先安装[requirements-onnx.txt](/requirements-onnx.txt))。

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

请注意，被输出的特征点已经换回原来的图像大小了。

您也可以运行[`infer.py`](/infer.py)。

```bash
python infer.py \
  --img_paths assets/DSC_0410.JPG assets/DSC_0411.JPG \
  --img_size 512 \
  --lightglue_path weights/superpoint_lightglue.onnx \
  --extractor_type superpoint \
  --extractor_path weights/superpoint.onnx \
  --viz
```

## TensorRT

TensorRT推理使用ONNXRuntime的TensorRT Execution Provider。请先安装[TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)。

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
  --lightglue_path  weights/superpoint_lightglue.onnx \
  --extractor_type superpoint \
  --extractor_path weights/superpoint.onnx \
  --trt \
  --viz
```

第一次运行时，TensorRT需要一点时间始化`.engine`和`.profile`。后续运行应使用cache。请注意，ONNX模型不应使用`--mp`或`--flash`转换。只支持SuperPoint特征提取。

## 推理时间比较

一般来说，对于比较少数量的特征点，ONNX的速度和PyTorch的差不多。但是，随着特征点数量的增加，在CUDA上PyTorch的速度更快，而在CPU上ONNX的推理速度更快。请参阅[EVALUATION.md](/evaluation/EVALUATION.md)。

<p align="center"><a href="https://github.com/fabio-sim/LightGlue-ONNX/blob/main/evaluation/EVALUATION.md"><img src="../assets/latency.png" alt="Latency Comparison" width=80%></a>

## 引用

如果使用这项目的代码或想法，请引用原本作者的论文：[LightGlue](https://arxiv.org/abs/2306.13643)，[SuperPoint](https://arxiv.org/abs/1712.07629)，[DISK](https://arxiv.org/abs/2006.13566)。最后，如果本项目的ONNX版以任何方式帮助了您，也请给个star。

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
