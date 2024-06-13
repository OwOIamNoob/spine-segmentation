import monai
import numpy
import nibabel
import SimpleITK as sitk
import matplotlib.pyplot as plt

def mask_overlay(image, mask, colormap="gray"):
    