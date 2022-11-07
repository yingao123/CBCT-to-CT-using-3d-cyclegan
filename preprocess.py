"""将ct和cbct图像切割成固定大小的patch，作为网络的输入"""


import os
import numpy as np
import SimpleITK as sitk


def preprocess(input_path, output_path, patch_size=(96,96,5), overlap=(26,26,2)):
    patch_x, patch_y, patch_z = patch_size
    func1 = lambda x,y:(x[i]-y[i] for i in range(3))
    stride_x, stride_y, stride_z = func1(patch_size, overlap)
    name_list = os.listdir(os.path.join(input_path, 'ct'))
    for file in name_list:
        ct = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(input_path, 'ct', file)))
        cbct = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(input_path, 'cbct', file)))
        z_dim,y_dim,x_dim=ct.shape
        for z_s in range(0,z_dim,stride_z):
            for y_s in range(0,y_dim,stride_y):
                for x_s in range(0, x_dim, stride_x):
                    z_s = min(z_dim-patch_z, z_s)
                    y_s = min(y_dim - patch_y, y_s)
                    x_s = min(x_dim - patch_x, x_s)
                    ct_patch=ct[z_s:z_s+patch_z, y_s:y_s+patch_y, x_s:x_s+patch_x]
                    cbct_patch = cbct[z_s:z_s + patch_z, y_s:y_s + patch_y, x_s:x_s + patch_x]
                    identity = '_' + str(z_s)+'_'+str(y_s)+'_' + str(x_s)
                    np.save(os.path.join(output_path, 'ct', file + identity+'.npy'), ct_patch)
                    np.save(os.path.join(output_path, 'cbct', file + identity+'.npy'), cbct_patch)


if __name__=='__main__':
    input_path = 'path_to_raw_data'
    cbct_path = os.path.join(input_path, 'cbct')
    ct_path = os.path.join(input_path, 'ct')
    output_path = 'path_to_preprocessed_data'
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'cbct'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'ct'), exist_ok=True)
    preprocess(input_path, output_path)