# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

# Adapted by Remi Pautrat, Philipp Lindenberger
# Adapted by Fabio Milentiansen Sim

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


def simple_nms(scores, nms_radius: int):
    """Fast Non-maximum suppression to remove nearby points"""
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    scores = scores[None]  # max_pool bug
    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)[0]


@torch.jit.script_if_tracing
def top_k_keypoints(
    keypoints: torch.Tensor, scores: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if k >= keypoints.shape[0]:
        return keypoints, scores
    else:
        scores, indices = torch.topk(scores, k, dim=0, sorted=True)
        return keypoints[indices], scores

def bilinear_grid_sample(im, grid, align_corners=False):
    """Given an input and a flow-field grid, computes the output using input
    values and pixel locations from grid. Supported only bilinear interpolation
    method to sample the input pixels.

    Args:
        im (torch.Tensor): Input feature map, shape (N, C, H, W)
        grid (torch.Tensor): Point coordinates, shape (N, Hg, Wg, 2)
        align_corners {bool}: If set to True, the extrema (-1 and 1) are
            considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.
    Returns:
        torch.Tensor: A tensor with sampled points, shape (N, C, Hg, Wg)
    """
    n, c, h, w = im.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.view(n, -1)
    y = y.view(n, -1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # Apply default for grid_sample function zero padding
    im_padded = F.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0)
    padded_h = h + 2
    padded_w = w + 2
    # save points positions after padding
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

    # Clip coordinates to padded image size
    # gpu
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # x0 = torch.where(x0 < 0, torch.tensor(0).to(device), x0)
    # x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1).to(device), x0)
    # x1 = torch.where(x1 < 0, torch.tensor(0).to(device), x1)
    # x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1).to(device), x1)
    # y0 = torch.where(y0 < 0, torch.tensor(0).to(device), y0)
    # y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1).to(device), y0)
    # y1 = torch.where(y1 < 0, torch.tensor(0).to(device), y1)
    # y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1).to(device), y1)

    # cpu
    x0 = torch.where(x0 < 0, torch.tensor(0), x0)
    x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1), x0)
    x1 = torch.where(x1 < 0, torch.tensor(0), x1)
    x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1), x1)
    y0 = torch.where(y0 < 0, torch.tensor(0), y0)
    y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1), y0)
    y1 = torch.where(y1 < 0, torch.tensor(0), y1)
    y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1), y1)

    im_padded = im_padded.view(n, c, -1)

    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)

    Ia = torch.gather(im_padded, 2, x0_y0)
    Ib = torch.gather(im_padded, 2, x0_y1)
    Ic = torch.gather(im_padded, 2, x1_y0)
    Id = torch.gather(im_padded, 2, x1_y1)

    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)

def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints_x = torch.div(keypoints[..., 0], (w * s - s / 2 - 0.5))
    keypoints_y = torch.div(keypoints[..., 1], (h * s - s / 2 - 0.5))
    keypoints = torch.stack((keypoints_x, keypoints_y), dim=-1)

    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    # descriptors = torch.nn.functional.grid_sample(
    #     descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=True
    # )
    descriptors = bilinear_grid_sample(descriptors, keypoints.view(b, 1, -1, 2))
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """

    default_config = {
        "descriptor_dim": 48,
        "nms_radius": 4,
        "max_num_keypoints": None,
        "detection_threshold": 0.0005,
        "remove_borders": 4,
    }

    def __init__(self, **conf):
        super().__init__()
        self.conf = {**self.default_config, **conf}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.conf["descriptor_dim"], kernel_size=1, stride=1, padding=0
        )

        # url = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth"
        # self.load_state_dict(torch.hub.load_state_dict_from_url(url))

        mk = self.conf["max_num_keypoints"]
        if mk is not None and mk <= 0:
            raise ValueError('"max_num_keypoints" must be positive or None')

        print("Loaded SuperPoint model")

    def forward(
        self,
        image: torch.Tensor,  # (1, 1, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute keypoints, scores, descriptors for image"""
        # Shared Encoder
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        _, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(1, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(1, h * 8, w * 8)
        scores = simple_nms(scores, self.conf["nms_radius"])

        # scores.shape == (B, H, W)

        # Discard keypoints near the image borders
        if pad := self.conf["remove_borders"]:
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1

        # Below this, B > 1 is not supported as each image can have a different number of keypoints.

        # Extract keypoints
        best_kp = torch.where(scores > self.conf["detection_threshold"])
        scores = scores[best_kp]

        keypoints = torch.stack(best_kp[1:3], dim=-1)
        # keypoints.shape == (N, 2)

        # scores.shape == (N)

        # Keep the k keypoints with highest score
        if self.conf["max_num_keypoints"] is not None:
            keypoints, scores = top_k_keypoints(
                keypoints, scores, self.conf["max_num_keypoints"]
            )

        # Convert (h, w) to (x, y)
        keypoints = torch.flip(keypoints, (1,))

        # keypoints.shape == (N, 2)

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = sample_descriptors(keypoints, descriptors, 8).permute(0, 2, 1)

        # Insert artificial batch dimension
        return (
            keypoints[None],  # (1, N, 2)
            scores[None],  # (1, N)
            descriptors,  # (1, N, desc_dim)
        )
