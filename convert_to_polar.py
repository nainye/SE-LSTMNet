import nibabel as nib
import numpy as np
import os
import utils
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--PATH', type=str, default='/workspace/Dataset/COSMOS2022/Training_dataset_nifti/', help='NIfTI image path')
parser.add_argument('--save_PATH', type=str, default='/workspace/Conference/2022_KIICE(한국정보통신학회 추계)/dataset/gt_seg_polar/train/', help='path for saving polar conversion result')
parser.add_argument('--r', type=int, default=32, help='radius of polar patch')
parser.add_argument('--d', type=int, default=32, help='num of angles of polar patch')
parser.add_argument('--mode', type=str, default='None', help='augmentation mode (shift/low/None)')



def main():
    args = parser.parse_args()

    polar_PATH = args.PATH+'/polar_img/'
    cls_PATH = args.PATH+'/cls_img/'
    cts_PATH = args.PATH+'/cts_img/'
    roi_PATH = args.PATH+'/roi_img/'

    if not os.path.isdir(polar_PATH):
        os.mkdir(polar_PATH)
    if not os.path.isdir(cls_PATH):
        os.mkdir(cls_PATH)
    if not os.path.isdir(cts_PATH):
        os.mkdir(cts_PATH)
    if not os.path.isdir(roi_PATH):
        os.mkdir(roi_PATH)

    arteries = ['L', 'R']
    pats = os.listdir(args.PATH)

    for p in pats:
        for a in arteries:
            print("Start!,,,", p, a)
            img_path = args.PATH+p+"/vista.nii.gz"

            lumen_path = args.PATH+p+"/"+a+"_"+"lumen_area.nii.gz"
            outer_path = args.PATH+p+"/"+a+"_"+"outer_area.nii.gz"
            vessel_path = args.PATH+p+"/"+a+"_"+"vessel_area.nii.gz"

            img_arr = nib.load(img_path).get_fdata()
            lumen_arr = nib.load(lumen_path).get_fdata()
            outer_arr = nib.load(outer_path).get_fdata()
            vessel_arr = nib.load(vessel_path).get_fdata()

            slices = sorted(list(set(np.where(lumen_arr!=0)[2])))

            for s in slices:
                img = img_arr[:,:,s]
                lumen = lumen_arr[:,:,s]
                outer = outer_arr[:,:,s]
                vessel = vessel_arr[:,:,s]

                lumen_points = np.stack([np.where(lumen==1)[0], np.where(lumen==1)[1]], axis=1)
                center = np.round(utils.centeroid(lumen_points)).astype(int)

                if args.mode == 'shift':
                    shift = [(2,0), (-1,0), (0,2), (0,-1)]
                else:
                    shift = [(0,0)]

                for sx, sy in shift:
                    center_coor = center.copy()
                    center_coor[0] = center_coor[0]+sx
                    center_coor[1] = center_coor[1]+sy

                    cartesian_patch = utils.Crop(img, center_coor, (args.r*2,args.r*2))
                    cartesian_lumen_patch = utils.Crop(lumen, center_coor, (args.r*2,args.r*2))
                    cartesian_outer_patch = utils.Crop(outer, center_coor, (args.r*2,args.r*2))
                    cartesian_vessel_patch = utils.Crop(vessel, center_coor, (args.r*2,args.r*2))

                    if np.count_nonzero(cartesian_lumen_patch) < 1:
                        continue

                    if args.mode == 'low':
                        cartesian_patch = cv2.resize(cartesian_patch, (args.r, args.r), interpolation=cv2.INTER_CUBIC)
                        cartesian_patch = cv2.resize(cartesian_patch, (args.r*2, args.r*2), interpolation=cv2.INTER_CUBIC)

                    if args.mode == 'shift':
                        utils.np2nii(cartesian_patch).to_filename(cts_PATH+p+"_"+str(a)+"_"+str(s)+"_"+str(sx)+str(sy)+".nii.gz")
                        utils.np2nii(cartesian_vessel_patch).to_filename(roi_PATH+p+"_"+str(a)+"_"+str(s)+"_"+str(sx)+str(sy)+".nii.gz")
                    elif args.mode == 'low':
                        utils.np2nii(cartesian_patch).to_filename(cts_PATH+p+"_"+str(a)+"_"+str(s)+"_low.nii.gz")
                        utils.np2nii(cartesian_vessel_patch).to_filename(roi_PATH+p+"_"+str(a)+"_"+str(s)+"_low.nii.gz")
                    else:
                        utils.np2nii(cartesian_patch).to_filename(cts_PATH+p+"_"+str(a)+"_"+str(s)+".nii.gz")
                        utils.np2nii(cartesian_vessel_patch).to_filename(roi_PATH+p+"_"+str(a)+"_"+str(s)+".nii.gz")

                    polar_patch = utils.topolar(cartesian_patch, rsamples=args.r, thsamples=args.d, intmethod='cubic')
                    polar_lumen_patch = utils.topolar(cartesian_lumen_patch, rsamples=args.r, thsamples=args.d, intmethod='cubic')
                    polar_outer_patch = utils.topolar(cartesian_outer_patch, rsamples=args.r, thsamples=args.d, intmethod='cubic')
                    polar_vessel_patch = utils.topolar(cartesian_vessel_patch, rsamples=args.r, thsamples=args.d, intmethod='cubic')

                    polar_lumen_patch[polar_lumen_patch>0.5] = 1
                    polar_lumen_patch[polar_lumen_patch<0.5] = 0
                    polar_outer_patch[polar_outer_patch>0.5] = 1
                    polar_outer_patch[polar_outer_patch<0.5] = 0
                    polar_vessel_patch[polar_vessel_patch>0.5] = 1
                    polar_vessel_patch[polar_vessel_patch<0.5] = 0

                    polar_lumen = polar_patch.copy()
                    polar_lumen[polar_lumen_patch!=1] = polar_lumen.min()
                    polar_outer = polar_patch.copy()
                    polar_outer[polar_outer_patch!=1] = polar_outer.min()
                    polar_vessel = polar_patch.copy()
                    polar_vessel[polar_vessel_patch!=1] = polar_vessel.min()

                    total_patches = np.stack([polar_patch, polar_lumen, polar_outer, polar_vessel], axis=2)

                    if args.mode == 'shift':
                        print(polar_PATH+p+"_"+str(a)+"_"+str(s)+"_"+str(sx)+str(sy)+".nii.gz")
                        utils.np2nii(polar_patch).to_filename(polar_PATH+p+"_"+str(a)+"_"+str(s)+"_"+str(sx)+str(sy)+".nii.gz")
                        utils.np2nii(total_patches).to_filename(cls_PATH+p+"_"+str(a)+"_"+str(s)+"_"+str(sx)+str(sy)+".nii.gz")
                    if args.mode == 'low':
                        print(polar_PATH+p+"_"+str(a)+"_"+str(s)+"_low.nii.gz")
                        utils.np2nii(polar_patch).to_filename(polar_PATH+p+"_"+str(a)+"_"+str(s)+"_low.nii.gz")
                        utils.np2nii(total_patches).to_filename(cls_PATH+p+"_"+str(a)+"_"+str(s)+"_low.nii.gz")
                    else:
                        print(polar_PATH+p+"_"+str(a)+"_"+str(s)+"_low.nii.gz")
                        utils.np2nii(polar_patch).to_filename(polar_PATH+p+"_"+str(a)+"_"+str(s)+".nii.gz")
                        utils.np2nii(total_patches).to_filename(cls_PATH+p+"_"+str(a)+"_"+str(s)+".nii.gz")



if __name__ == "__main__":
    main()