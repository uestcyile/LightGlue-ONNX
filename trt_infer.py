"""Sample code to build and run LightGlue TensorRT engine inference."""
import numpy as np
import tensorrt as trt  # >= 8.6.1
import torch

import trt_utils.common as common
from lightglue_onnx import SuperPoint
from lightglue_onnx.end2end import normalize_keypoints
from lightglue_onnx.utils import load_image, rgb_to_grayscale


def build_engine(
    model_path: str, output_path: str, num_keypoints: int = 512, desc_dim: int = 256
):
    logger = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(logger)

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    parser = trt.OnnxParser(network, logger)

    success = parser.parse_from_file(model_path)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        raise Exception

    config = builder.create_builder_config()

    profile = builder.create_optimization_profile()

    for name in ["kpts0", "kpts1"]:
        profile.set_shape(
            name,
            (1, 32, 2),
            (1, num_keypoints // 2, 2),
            (1, num_keypoints, 2),
        )
    for name in ["desc0", "desc1"]:
        profile.set_shape(
            name,
            (1, 32, desc_dim),
            (1, num_keypoints // 2, desc_dim),
            (1, num_keypoints, desc_dim),
        )

    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)

    with open(output_path, "wb") as f:
        f.write(serialized_engine)


def load_inputs(
    input_buffers,
    img_size=512,
    img0_path="assets/sacre_coeur1.jpg",
    img1_path="assets/sacre_coeur2.jpg",
    max_num_keypoints=512,
):
    image0, scales0 = load_image(img0_path, resize=img_size)
    image1, scales1 = load_image(img1_path, resize=img_size)
    image0 = rgb_to_grayscale(image0)
    image1 = rgb_to_grayscale(image1)
    extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval()

    with torch.inference_mode():
        feats0, feats1 = extractor(image0[None]), extractor(image1[None])
        kpts0, scores0, desc0 = feats0
        kpts1, scores1, desc1 = feats1

        kpts0 = normalize_keypoints(kpts0, image0.shape[1], image0.shape[2])
        kpts1 = normalize_keypoints(kpts1, image1.shape[1], image1.shape[2])

    for i, tensor in zip(input_buffers, [kpts0, kpts1, desc0, desc1]):
        np.copyto(i.host, tensor.numpy().ravel())

    return {
        "kpts0": kpts0.shape,
        "kpts1": kpts1.shape,
        "desc0": desc0.shape,
        "desc1": desc1.shape,
    }


def run_engine(engine_path: str):
    logger = trt.Logger(trt.Logger.WARNING)

    with open(engine_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    # TODO: Dynamic output shapes
    inputs, outputs, bindings, stream = common.allocate_buffers(engine, profile_idx=0)

    shapes = load_inputs(inputs)
    context = engine.create_execution_context()

    for name, shape in shapes.items():
        context.set_input_shape(name, tuple(shape))

    trt_outputs = common.do_inference_v2(
        context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
    )

    matches0, mscores0 = trt_outputs

    return matches0, mscores0


if __name__ == "__main__":
    model_path = "weights/superpoint_lightglue.trt.onnx"
    output_path = "weights/superpoint_lightglue.engine"

    build_engine(model_path, output_path)
    matches0, mscores0 = run_engine(output_path)

    print(matches0.reshape(512, 2))
