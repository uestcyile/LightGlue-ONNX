import argparse

import torch

from lightglue_onnx import LightGlue, SuperPoint
from lightglue_onnx.utils import load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=512, required=False)
    parser.add_argument(
        "--superpoint_path", type=str, default="weights/superpoint.onnx", required=False
    )
    parser.add_argument(
        "--lightglue_path",
        type=str,
        default="weights/superpoint_lightglue.onnx",
        required=False,
    )
    return parser.parse_args()


def export_onnx(
    img_size=512,
    superpoint_path="weights/superpoint.onnx",
    lightglue_path="weights/superpoint_lightglue.onnx",
    img0_path="assets/sacre_coeur1.jpg",
    img1_path="assets/sacre_coeur2.jpg",
):
    # Sample images for tracing
    image0, scales0 = load_image(img0_path, resize=img_size)
    image1, scales1 = load_image(img1_path, resize=img_size)

    # Models
    extractor = SuperPoint().eval()
    matcher = LightGlue("superpoint").eval()

    # Export Superpoint
    torch.onnx.export(
        extractor,
        image0[None],
        superpoint_path,
        input_names=["image"],
        output_names=["keypoints", "scores", "descriptors"],
        opset_version=16,
        dynamic_axes={
            "image": {2: "height", 3: "width"},
            "keypoints": {0: "num_keypoints"},
            "scores": {0: "num_scores"},
            "descriptors": {0: "num_descriptors"},
        },
    )

    # Export LightGlue
    feats0, feats1 = extractor(image0[None]), extractor(image1[None])
    kpts0, scores0, desc0 = feats0
    kpts1, scores1, desc1 = feats1

    torch.onnx.export(
        matcher,
        (
            kpts0[None],
            kpts1[None],
            desc0[None],
            desc1[None],
            image0[None],
            image1[None],
        ),
        lightglue_path,
        input_names=["kpts0", "kpts1", "desc0", "desc1", "image0", "image1"],
        output_names=["matches0", "matches1", "mscores0", "mscores1"],
        opset_version=16,
        dynamic_axes={
            "kpts0": {1: "num_kpts0"},
            "kpts1": {1: "num_kpts1"},
            "desc0": {1: "num_desc0"},
            "desc1": {1: "num_desc1"},
            "image0": {2: "height_0", 3: "width_0"},
            "image1": {2: "height_1", 3: "width_1"},
        },
    )


if __name__ == "__main__":
    args = parse_args()
    export_onnx(**vars(args))
