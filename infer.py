from onnx_runner import SuperpointLightglueRunner, load_image

img0_path = "assets/sacre_coeur1.jpg"
img1_path = "assets/sacre_coeur2.jpg"
size = 512
image0, scales0 = load_image(img0_path, resize=size)
image1, scales1 = load_image(img1_path, resize=size)

# Create ONNXRuntime runner
runner = SuperpointLightglueRunner(
    superpoint_path="weights/superpoint.onnx",
    lightglue_path="weights/superpoint_lightglue.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

# Run inference
m_kpts0, m_kpts1 = runner.run(image0, image1, scales0, scales1)
