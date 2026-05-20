import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from time import time
import random


def SingleAnchorDPR(centerx, centery, r, num_regions, h, w, vis=False, cc=-100):
    centerx = centerx.cuda()
    centery = centery.cuda()
    r = r.cuda()
    assert r.shape[1] == num_regions, 'the len of r should be equal with num_regions.'
    bs = r.shape[0]
    # Get the matrix of coordinate
    coordinate = torch.zeros((bs, h, w, 2)).cuda()
    coordinate[:, :, :, 0] = (torch.arange(h).repeat(w, 1)).T.cuda()
    coordinate[:, :, :, 1] = torch.arange(w).repeat(h, 1).cuda()
    # Generate the mask of regions
    relate_cx = coordinate[:, :, :, 0] - centerx.view(bs, 1, 1)
    relate_cy = coordinate[:, :, :, 1] - centery.view(bs, 1, 1)
    mask_region = (torch.acos(relate_cy / (torch.sqrt(relate_cx ** 2 + relate_cy ** 2))) / np.pi * 180).cuda()
    dis = torch.sqrt(relate_cx ** 2 + relate_cy ** 2).cuda()

    sub_angle_index = (relate_cx < 0)
    mask_region[sub_angle_index] = 2 * 180 - mask_region[sub_angle_index]
    mask_region = mask_region // (360 // num_regions)
    mask_region[range(bs), centerx, centery] = num_regions
    mask_region = mask_region.long()

    # Calculate points
    points = torch.zeros((bs, num_regions, 2)).cuda()  # [bs, num_regions, 2]
    act = torch.zeros((bs, h, w)).cuda()
    for i in range(num_regions):
        angle = (360 / num_regions * i / 360 * 2 * torch.Tensor([np.pi])).cuda()
        points[:, i, 0] = centerx * 1.0 + r[:, i] * torch.sin(angle)
        points[:, i, 1] = centery * 1.0 + r[:, i] * torch.cos(angle)

    # Calculate Act
    for i in range(num_regions):
        idx = torch.nonzero(mask_region == i)
        if (num_regions - 1) == i:
            a = points[:, 0]
            b = points[:, i]
        else:
            a = points[:, i]
            b = points[:, i + 1]
        A, B, C = GaussianElimination(a, b)

        bs_idx = idx[:, 0]
        c = idx[:, 1:]
        tx = centerx[bs_idx]
        ty = centery[bs_idx]
        o = torch.stack((tx, ty), dim=1).float()
        A1, B1, C1 = GaussianElimination(o, c.float())

        A0 = A[bs_idx]
        B0 = B[bs_idx]
        C0 = C[bs_idx]

        D = A0 * B1 - A1 * B0
        x = (B0 * C1 - B1 * C0) * 1.0 / D
        y = (A1 * C0 - A0 * C1) * 1.0 / D

        assert torch.isnan(x).long().sum() == 0, 'Calculate act has been found None!'

        before_act = dis[bs_idx, c[:, 0], c[:, 1]] / torch.sqrt(
            (1.0 * o[:, 0] - x) ** 2 + (1.0 * o[:, 1] - y) ** 2)
        act[bs_idx, c[:, 0], c[:, 1]] = ActFunc(before_act, cc=cc)

    act[range(bs), centerx, centery] = 1

    if vis:
        for i in range(10):
            point = points[i].detach().cpu().numpy().copy()
            # 将 np.int 替换为 np.int32 或 int
            point = np.array(point).astype(np.int32)[:, ::-1]
            tmp_mask_region = mask_region[i].reshape(h, w, 1).cpu().numpy().astype(np.uint8)
            tmp_mask_region = cv2.fillPoly(img=tmp_mask_region, pts=[point], color=40)
            plt.imshow(tmp_mask_region.reshape(h, w))
            plt.colorbar()
            plt.show()
            t_act = act[i].detach().cpu().numpy()
            plt.imshow(t_act)
            plt.colorbar()
            plt.show()

    return act

def ActFunc(x, cc=-100):
    ans = (torch.tanh(cc * (x - 1)) + 1) / 2
    return ans

def GaussianElimination(a, b):
    # Ax+By+C=0
    first_x, first_y, second_x, second_y = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    A = 1.0 * second_y - 1.0 * first_y
    B = 1.0 * first_x - 1.0 * second_x
    C = 1.0 * second_x * first_y - 1.0 * first_x * second_y
    return A, B, C


# 随机位置选择
def random_location(self, n=1):
    assert n >= 1
    assert len(self.allowed_pixels) > 0
    start_pixels = random.choices(tuple(self.allowed_pixels), k=n)
    return np.array([(y, x, self.mask_dims[0], self.mask_dims[1]) for (y, x) in start_pixels])

# 掩码粘贴
def get_masks(self, mask_coords, n_channels):
    assert n_channels >= 1
    batch_size = len(mask_coords)
    masks = np.zeros(
        (batch_size, n_channels, self.img_dims[0], self.img_dims[1]), dtype=np.float32)
    for b in range(batch_size):
        masks[b, :, mask_coords[b][0]:mask_coords[b][0]+self.mask_dims[0],
              mask_coords[b][1]:mask_coords[b][1]+self.mask_dims[1]] = 1
    return masks


