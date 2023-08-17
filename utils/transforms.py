#  Author: fengping su
#  date: 2023-8-16
#  All rights reserved.
import cv2
import numpy as np
import torch

# implementation of filter bank preprocess module using DCT
def filter_bank_preprocess(img_path: str, k: int=10):
    img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_dct = cv2.dct(img)
    img_dct_f = np.abs(img_dct)

    low_freq = img_dct_f > 2*k
    mid_freq = (img_dct_f >= k) * (img_dct_f < 2*k)
    high_freq = img_dct_f < k

    img_dct_low = cv2.idct(img_dct * low_freq)[:, :, None]
    img_dct_mid = cv2.idct(img_dct * mid_freq)[:, :, None]
    img_dct_high = cv2.idct(img_dct * high_freq)[:, :, None]
    return torch.from_numpy(np.concatenate((img_dct_low, img_dct_mid, img_dct_high), axis=2))