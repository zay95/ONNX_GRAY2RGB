# -*- coding: utf-8 -*-
'''
@author: funny_QZQ
@file: test_onnx.py
@time: 2020/7/2 上午11:11
'''

import os, sys

from torchvision import transforms
from PIL import Image
import onnxruntime
# import onnxruntime_gpu
import torch
import numpy as np
import time


def get_transforms():

    def __resize(img, w=512, h=512, method=Image.BICUBIC):
        return img.resize((w, h), method)
    transform_list = list()
    transform_list.append(transforms.Lambda(lambda img: __resize(img, 256, 256)))

    # ToTensor
    transform_list += [transforms.ToTensor()]

    # Normalize
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def save_tensor_to_img(img_tensor, img_path):
    # tensor: 4D tensor
    img_tensor = img_tensor.detach().cpu().float().numpy()
    image_numpy = (np.transpose(img_tensor, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_pil = Image.fromarray(image_numpy.astype(np.uint8))

    # save to png
    image_pil.save(img_path)


if __name__ == '__main__':

    # onnx model
    ort_session = onnxruntime.InferenceSession("G_8_gray2rgb_256.onnx")
    # print(onnxruntime_gpu.get_device())

    transformer = get_transforms()

    test_img_folder = 'test_images_gray'
    img_names = os.listdir(test_img_folder)

    res_folder = 'imgs_show_RGB'
    if not os.path.exists(res_folder): os.makedirs(res_folder)

    for img_index, img_name in enumerate(img_names):
        img_path = os.path.join(test_img_folder, img_name)

        # img_path = './test_images/000000.png'

        img = Image.open(img_path)
        img = img.convert("RGB")
        trans_img = transformer(img)
        trans_img.unsqueeze_(0)

        onnx_inputs = {ort_session.get_inputs()[0].name: to_numpy(trans_img)}

        start_time_2 = time.time()
        onnx_outs = ort_session.run(None, onnx_inputs)
        print('onnx_model time consume: {} s'.format(time.time() - start_time_2))

        onnx_out = torch.FloatTensor(onnx_outs[0])

        # save image
        save_tensor_to_img(onnx_out[0], os.path.join(res_folder, 'onnx_model_out_{}.png'.format(img_index)))

