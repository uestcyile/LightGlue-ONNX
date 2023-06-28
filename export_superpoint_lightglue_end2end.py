import argparse

import torch

from lightglue_onnx import SuperPointLightGlueEnd2End
from lightglue_onnx.utils import load_image, rgb_to_grayscale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=512, required=False)
    parser.add_argument(
        "--save_path",
        type=str,
        default="weights/superpoint_lightglue_end2end.onnx",
        required=False,
    )
    return parser.parse_args()


def export_onnx(
    img_size=512,
    save_path="weights/superpoint_lightglue_end2end.onnx",
    img0_path="assets/sacre_coeur1.jpg",
    img1_path="assets/sacre_coeur2.jpg",
):
    # Sample images for tracing
    image0, scales0 = load_image(img0_path, resize=img_size)
    image1, scales1 = load_image(img1_path, resize=img_size)
    image0 = rgb_to_grayscale(image0)
    image1 = rgb_to_grayscale(image1)

    # Models
    model = SuperPointLightGlueEnd2End()

    # Export End2End
    torch.onnx.export(
        model,
        (image0[None], image1[None]),
        save_path,
        input_names=["image0", "image1"],
        output_names=["kpts0", "kpts1", "matches0", "matches1", "mscores0", "mscores1"],
        opset_version=16,
        dynamic_axes={
            "image0": {2: "height0", 3: "width0"},
            "image1": {2: "height1", 3: "width1"},
            "kpts0": {1: "num_keypoints0"},
            "kpts1": {1: "num_keypoints1"},
            "matches0": {1: "num_matches0"},
            "matches1": {1: "num_matches1"},
            "mscores0": {1: "num_matches0"},
            "mscores1": {1: "num_matches1"},
        },
    )


if __name__ == "__main__":
    args = parse_args()
    export_onnx(**vars(args))
