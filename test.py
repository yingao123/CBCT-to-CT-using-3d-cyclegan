import os
from options.test_options import TestOptions
from data import create_dataset
from models import networks
import torch
import SimpleITK as sitk
import numpy as np


def test_merge(model, input_path, save_path, patch_size = (96, 96, 5), overlap = (26, 26, 2)):
    """逐patch做测试，重叠区域取平均"""
    patch_x, patch_y, patch_z = patch_size
    func1 = lambda x, y: (x[i] - y[i] for i in range(3))
    stride_x, stride_y, stride_z = func1(patch_size, overlap)
    real_cbct_itk = sitk.ReadImage(input_path)
    real_cbct = sitk.GetArrayFromImage(real_cbct_itk)
    syn_ct = np.zeros_like(real_cbct)
    counts = np.zeros_like(real_cbct)
    z_dim, y_dim, x_dim = real_cbct.shape
    for z_s in range(0, z_dim, stride_z):
        for y_s in range(0, y_dim, stride_y):
            for x_s in range(0, x_dim, stride_x):
                z_s = min(z_dim - patch_z, z_s)
                y_s = min(y_dim - patch_y, y_s)
                x_s = min(x_dim - patch_x, x_s)
                real_cbct_patch = real_cbct[z_s:z_s + patch_z, y_s:y_s + patch_y, x_s:x_s + patch_x]
                syn_ct_patch = model(real_cbct_patch)
                syn_ct[z_s:z_s + patch_z, y_s:y_s + patch_y, x_s:x_s + patch_x]+=syn_ct_patch
                counts[z_s:z_s + patch_z, y_s:y_s + patch_y, x_s:x_s + patch_x]+=1
    syn_ct/=counts
    ### 还需要归一化回去？目前结果为0均值、1方差
    syn_ct_itk = sitk.GetImageFromArray(syn_ct)
    syn_ct_itk.CopyInformation(real_cbct_itk)
    filename = input_path.split('/')[-1]
    sitk.WriteImage(syn_ct_itk, os.path.join(save_path, filename))


if __name__ == '__main__':
    load_path = 'path to model parameters'
    realA_path = 'path to cbct path'
    synB_path = 'path to ct path'
    opt = TestOptions().parse()  # get test options
    model = networks.UnetGenerator(input_nc=opt.input_nc, output_nc=opt.output_nc, num_downs=4, ngf=opt.ngf)      # create a model given opt.model and other options

    state_dict = torch.load(load_path, map_location={'cuda:0'})
    model.load_state_dict(state_dict)

    model.eval()
    for filename in os.listdir(realA_path):
        test_merge(model, os.path.join(realA_path, filename), synB_path)
