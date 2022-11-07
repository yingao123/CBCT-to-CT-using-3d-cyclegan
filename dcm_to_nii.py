import SimpleITK as sitk
import numpy as np
import os,glob
import sys
import pydicom
import argparse
parser = argparse.ArgumentParser()
parser.description='please enter two parameters a and b ...'
parser.add_argument("-o", "--output_path", help="path to output nii files",  type=str,
                    default=r'F:\Rectal\Data\yangmw_nii')
parser.add_argument("-i", "--input_path", help="path to input dicom files",  type=str,
                    default=r'F:\Rectal\Data\yangmw')
args = parser.parse_args()

output_path=args.output_path
os.makedirs(output_path,exist_ok=True)
input_path=args.input_path
reader = sitk.ImageSeriesReader()
list = []
for x in os.listdir(output_path):
    list.append(x.split('_')[0])
for patient_name in os.listdir(input_path):
    dicom_path = os.path.join(input_path, patient_name)
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    # if image.GetSize()[-1] <= 10:
    #     continue
    adicom = os.listdir(dicom_path)
    adicom = adicom[0]
    try:
        ds = pydicom.read_file(os.path.join(dicom_path, adicom))
        date = ds['StudyDate'].value
        pid = ds['PatientID'].value
        age = int(ds['PatientAge'].value[:-1])
        sex = ds['PatientSex'].value
    except:
        try:
            age = int(ds['StudyDate'].value[:4]) - int(ds['PatientBirthDate'].value[:4])
            sex = ds['PatientSex'].value
        except:
            continue
    output_name = os.path.join(output_path, patient_name + '.nii')
    print(output_name)
    sitk.WriteImage(image,output_name)



