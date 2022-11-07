import numpy as np
import math
from interpolation.splines import UCGrid
from interpolation.splines import eval_linear
from interpolation.splines import filter_cubic, eval_cubic

def get_start_and_margin(target_size, img_size, center):
    if center - (target_size//2) < 0:
        start = 0
        margin = (target_size//2) - center
    else:
        start = center - (target_size//2)
        margin = 0
    return start, margin

def get_end_and_margin(target_size, img_size, center):
    if center + (target_size//2) > img_size:
        end = img_size
        margin = center + (target_size//2) - img_size
    else:
        end = center + (target_size//2)
        margin = 0
    return end, margin

def crop_2D_image(window, slice_arr, center):
    x_start, x_start_m = get_start_and_margin(window[0], slice_arr.shape[0], center[0])
    x_end, x_end_m = get_end_and_margin(window[0], slice_arr.shape[0], center[0])

    y_start, y_start_m = get_start_and_margin(window[1], slice_arr.shape[1], center[1])
    y_end, y_end_m = get_end_and_margin(window[1], slice_arr.shape[1], center[1])
    
    # print('Margins :', x_start_m, x_end_m, y_start_m, y_end_m)
    
    cropped_slice = slice_arr[x_start:x_end, y_start:y_end]
    
    if cropped_slice.shape != window:
        cropped_slice = np.pad(cropped_slice,
                               pad_width=((x_start_m, x_end_m), (y_start_m, y_end_m)),
                               mode='constant',
                               constant_values=0)
    
    # print("crop size :", cropped_slice.shape)
    
    return cropped_slice

def centeroid(data):
    x, y = zip(*data)
    l = len(x)
    return sum(x) / l, sum(y) / l

def Crop(img, center, size=(64,64)): 
    if len(img.shape) == 3:
        slice0 = crop_2D_image(size, img[:,:,0], center)
        slice1 = crop_2D_image(size, img[:,:,1], center)
        slice2 = crop_2D_image(size, img[:,:,2], center)
        return np.stack([slice0, slice1, slice2], axis=2)
    
    return crop_2D_image(size, img, center)

def np2nii(arr):
    import nibabel as nib
    import numpy as np
    if len(arr.shape) == 2:
        arr = arr[:,:,np.newaxis]
    img = nib.Nifti1Image(arr,np.eye(4))
    img.header.get_xyzt_units()
    return img


# From cartesian image patch to polar image patch
def topolar(car_img, rsamples = 0, thsamples = 256, intmethod = 'linear'):
    if rsamples==0:
        rsamples = car_img.shape[0]//2
    if len(car_img.shape)==2:
        cimg = car_img[:,:,None]
    elif len(car_img.shape)==3:
        cimg = car_img
    else:
        print('channel not 2/3')
        return

    SUBTH = 360/thsamples
    channel=cimg.shape

    grid = UCGrid((0, cimg.shape[1]-1, cimg.shape[1]), (0, cimg.shape[0]-1, cimg.shape[0]))

    if intmethod == 'cubic':
        coeffs = filter_cubic(grid, cimg) 
    
    rth = np.zeros((thsamples,rsamples,channel))
    for th in range(thsamples):
        for r in range(rsamples):
            inty = cimg.shape[0]//2+r*math.sin(th*SUBTH/180*math.pi)
            intx = cimg.shape[1]//2+r*math.cos(th*SUBTH/180*math.pi)
            if intx>=cimg.shape[1]-1 or inty>=cimg.shape[0]-1:
                rth[th,r] = 0
            elif intx<0 or inty<0:
                rth[th,r] = 0
            else:    
                if intmethod == 'cubic':
                    rth[th,r] = eval_cubic(grid, coeffs, np.array([inty,intx]))
                elif intmethod == 'linear':
                    rth[th,r] = eval_linear(grid, cimg, np.array([inty,intx]))

    if len(car_img.shape)==2:
        return rth[:,:,0]
    else:
        return rth
 