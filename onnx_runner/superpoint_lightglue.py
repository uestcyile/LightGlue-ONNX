import numpy as np
import onnxruntime as ort


class SuperpointLightglueRunner:
    def __init__(
        self,
        superpoint_path="weights/superpoint.onnx",
        lightglue_path="weights/superpoint_lightglue.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    ):
        self.superpoint = ort.InferenceSession(superpoint_path, providers=providers)
        self.lightglue = ort.InferenceSession(lightglue_path, providers=providers)

    def run(self, image0: np.ndarray, image1: np.ndarray, scales0, scales1):
        kpts0, scores0, desc0 = self.superpoint.run(None, {"image": image0})
        kpts1, scores1, desc1 = self.superpoint.run(None, {"image": image1})

        matches0, matches1, mscores0, mscores1 = self.lightglue.run(
            None,
            {
                "kpts0": kpts0[None],
                "kpts1": kpts1[None],
                "desc0": desc0[None],
                "desc1": desc1[None],
                "image0": image0,
                "image1": image1,
            },
        )
        m_kpts0, m_kpts1 = self.post_process(kpts0, kpts1, matches0, scales0, scales1)
        return m_kpts0, m_kpts1

    def post_process(self, kpts0, kpts1, matches0, scales0, scales1):
        kpts0 = (kpts0[None] + 0.5) / scales0[None] - 0.5
        kpts1 = (kpts1[None] + 0.5) / scales1[None] - 0.5
        # create match indices
        valid = matches0[0] > -1
        matches = np.stack([np.where(valid)[0], matches0[0][valid]], -1)
        m_kpts0, m_kpts1 = kpts0[0][matches[..., 0]], kpts1[0][matches[..., 1]]
        return m_kpts0, m_kpts1
