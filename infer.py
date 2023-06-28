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
