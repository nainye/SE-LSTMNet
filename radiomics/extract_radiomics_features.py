import radiomics
import SimpleITK as sitk
import os
import numpy as np
import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help='train/test')
args = parser.parse_args()

def main():
    if args.mode == 'train':
        fpath = "/workspace/Dataset/COSMOS2022/Training_dataset_nifti/"
        with open("slice_label.json", "r") as json_file:
            slicelabels = json.load(json_file)
        
    else:
        fpath = "/workspace/Dataset/COSMOS2022/Testing_dataset_nifti/"
        with open('test_label.json', "r") as json_file:
            slicelabels = json.load(json_file)
    
    Pats = os.listdir(fpath)

    FeatureStorage = []
    exceptslice = []

    arteries = ['L', 'R']
    for i, p in enumerate(Pats):
        print("PatNum:", p)
        
        imageName = fpath + p + "/vista.nii.gz"
        
        for a in arteries:
            if a == 'L':
                LumenName = fpath+p+"/L_lumen_area.nii.gz"
                OuterName = fpath+p+"/L_outer_area.nii.gz"
                VesselName = fpath+p+"/L_vessel_area.nii.gz"
            else:
                LumenName = fpath+p+"/R_lumen_area.nii.gz"
                OuterName = fpath+p+"/R_outer_area.nii.gz"
                VesselName = fpath+p+"/R_vessel_area.nii.gz"

            image = sitk.ReadImage(imageName)
            Lumen = sitk.ReadImage(LumenName)
            Outer = sitk.ReadImage(OuterName)
            Vessel = sitk.ReadImage(VesselName)

            lumen = (Lumen==1)
            outer = (Outer==1)
            vessel = (Vessel==1)

            anno_slice  = slicelabels[p][a]

            lumen_arr = sitk.GetArrayFromImage(lumen)
            anno_slice_seg = sorted(list(set(np.where(lumen_arr==1)[0])))

            for s in anno_slice:
                if int(s) not in anno_slice_seg:
                    continue

                storage = []
                storage.append(p)
                storage.append(a)
                storage.append(s)
                storage.append(slicelabels[p][a][s])
                
                s = int(s)

                LumenShapeFeatureExtractor = radiomics.shape2D.RadiomicsShape2D(image[:,:,s], lumen[:,:,s])
                LumenShapeFeatureExtractor.enableAllFeatures()
                LumenShapeFeatureExtractor.execute()

                OuterShapeFeatureExtractor = radiomics.shape2D.RadiomicsShape2D(image[:,:,s], outer[:,:,s])
                OuterShapeFeatureExtractor.enableAllFeatures()
                OuterShapeFeatureExtractor.execute()

                VesselShapeFeatureExtractor = radiomics.shape2D.RadiomicsShape2D(image[:,:,s], vessel[:,:,s])
                VesselShapeFeatureExtractor.enableAllFeatures()
                VesselShapeFeatureExtractor.execute()

                storage.append(LumenShapeFeatureExtractor.getMeshSurfaceFeatureValue())
                storage.append(LumenShapeFeatureExtractor.getPixelSurfaceFeatureValue())
                storage.append(LumenShapeFeatureExtractor.getPerimeterFeatureValue())
                storage.append(LumenShapeFeatureExtractor.getPerimeterSurfaceRatioFeatureValue())
                storage.append(LumenShapeFeatureExtractor.getSphericityFeatureValue())
                storage.append(LumenShapeFeatureExtractor.getSphericalDisproportionFeatureValue())
                storage.append(LumenShapeFeatureExtractor.getMaximumDiameterFeatureValue())
                storage.append(LumenShapeFeatureExtractor.getMajorAxisLengthFeatureValue())
                storage.append(LumenShapeFeatureExtractor.getMinorAxisLengthFeatureValue())
                storage.append(LumenShapeFeatureExtractor.getElongationFeatureValue())

                storage.append(OuterShapeFeatureExtractor.getMeshSurfaceFeatureValue())
                storage.append(OuterShapeFeatureExtractor.getPixelSurfaceFeatureValue())
                storage.append(OuterShapeFeatureExtractor.getPerimeterFeatureValue())
                storage.append(OuterShapeFeatureExtractor.getPerimeterSurfaceRatioFeatureValue())
                storage.append(OuterShapeFeatureExtractor.getSphericityFeatureValue())
                storage.append(OuterShapeFeatureExtractor.getSphericalDisproportionFeatureValue())
                storage.append(OuterShapeFeatureExtractor.getMaximumDiameterFeatureValue())
                storage.append(OuterShapeFeatureExtractor.getMajorAxisLengthFeatureValue())
                storage.append(OuterShapeFeatureExtractor.getMinorAxisLengthFeatureValue())
                storage.append(OuterShapeFeatureExtractor.getElongationFeatureValue())

                storage.append(VesselShapeFeatureExtractor.getMeshSurfaceFeatureValue())
                storage.append(VesselShapeFeatureExtractor.getPixelSurfaceFeatureValue())
                storage.append(VesselShapeFeatureExtractor.getPerimeterFeatureValue())
                storage.append(VesselShapeFeatureExtractor.getPerimeterSurfaceRatioFeatureValue())
                storage.append(VesselShapeFeatureExtractor.getSphericityFeatureValue())
                storage.append(VesselShapeFeatureExtractor.getSphericalDisproportionFeatureValue())
                storage.append(VesselShapeFeatureExtractor.getMaximumDiameterFeatureValue())
                storage.append(VesselShapeFeatureExtractor.getMajorAxisLengthFeatureValue())
                storage.append(VesselShapeFeatureExtractor.getMinorAxisLengthFeatureValue())
                storage.append(VesselShapeFeatureExtractor.getElongationFeatureValue())


                '''''''''''''''''''''''''''''''''''''''''''''
                # Histogram Based Features
                # Feature n = 14
                '''''''''''''''''''''''''''''''''''''''''''''

                print("Calculation histogram features...")

                settings = {'binWidth': 32, 'interpolator' : None, 'verbose' : True}

                # Lumen
                LumenHistFeatureExtractor = radiomics.firstorder.RadiomicsFirstOrder(image[:,:,s], lumen[:,:,s], **settings)
                LumenHistFeatureExtractor.enableAllFeatures()
                LumenHistFeatureExtractor.execute()

                # Outer
                OuterHistFeatureExtractor = radiomics.firstorder.RadiomicsFirstOrder(image[:,:,s], outer[:,:,s], **settings)
                OuterHistFeatureExtractor.enableAllFeatures()
                OuterHistFeatureExtractor.execute()

                # Vessel
                VesselHistFeatureExtractor = radiomics.firstorder.RadiomicsFirstOrder(image[:,:,s], vessel[:,:,s], **settings)
                VesselHistFeatureExtractor.enableAllFeatures()
                VesselHistFeatureExtractor.execute()


                '''''''''''''''''''''''''''''''''''''''''''''
                # texture Based Features (GLCM)
                # Feature n = 16
                '''''''''''''''''''''''''''''''''''''''''''''

                print("Calculation GLCM features...")

                settings = {'binWidth': 32, 'interpolator' : None, 'verbose' : True}        

                # Lumen
                LumenGLCMFeatureExtractor = radiomics.glcm.RadiomicsGLCM(image[:,:,s], lumen[:,:,s], **settings)
                LumenGLCMFeatureExtractor.enableAllFeatures()
                LumenGLCMFeatureExtractor.execute()

                # Outer
                OuterGLCMFeatureExtractor = radiomics.glcm.RadiomicsGLCM(image[:,:,s], outer[:,:,s], **settings)
                OuterGLCMFeatureExtractor.enableAllFeatures()
                OuterGLCMFeatureExtractor.execute()

                # Vessel
                VesselGLCMFeatureExtractor = radiomics.glcm.RadiomicsGLCM(image[:,:,s], vessel[:,:,s], **settings)
                VesselGLCMFeatureExtractor.enableAllFeatures()
                VesselGLCMFeatureExtractor.execute()

                '''''''''''''''''''''''''''''''''''''''''''''
                # texture Based Features (GLSZM)
                # Feature n = 16
                '''''''''''''''''''''''''''''''''''''''''''''

                print("Calculation GLSZM features...")

                settings = {'binWidth': 32, 'interpolator' : None, 'verbose' : True}        

                # Lumen
                LumenGLSZMFeatureExtractor = radiomics.glszm.RadiomicsGLSZM(image[:,:,s], lumen[:,:,s], **settings)
                LumenGLSZMFeatureExtractor.enableAllFeatures()
                LumenGLSZMFeatureExtractor.execute()

                # Outer
                OuterGLSZMFeatureExtractor = radiomics.glszm.RadiomicsGLSZM(image[:,:,s], outer[:,:,s], **settings)
                OuterGLSZMFeatureExtractor.enableAllFeatures()
                OuterGLSZMFeatureExtractor.execute()

                # Vessel
                VesselGLSZMFeatureExtractor = radiomics.glszm.RadiomicsGLSZM(image[:,:,s], vessel[:,:,s], **settings)
                VesselGLSZMFeatureExtractor.enableAllFeatures()
                VesselGLSZMFeatureExtractor.execute()


                # Save features in storage variable

                # Lumen
                storage.append(LumenHistFeatureExtractor.getMaximumFeatureValue()[0])
                storage.append(LumenHistFeatureExtractor.getMinimumFeatureValue()[0])
                storage.append(LumenHistFeatureExtractor.getMedianFeatureValue()[0])
                storage.append(LumenHistFeatureExtractor.getMeanFeatureValue()[0])
                storage.append(LumenHistFeatureExtractor.getVarianceFeatureValue()[0])
                storage.append(LumenHistFeatureExtractor.getEnergyFeatureValue()[0])
                storage.append(LumenHistFeatureExtractor.getStandardDeviationFeatureValue()[0])
                storage.append(LumenHistFeatureExtractor.getSkewnessFeatureValue()[0])
                storage.append(LumenHistFeatureExtractor.getKurtosisFeatureValue()[0])
                storage.append(LumenHistFeatureExtractor.getRootMeanSquaredFeatureValue()[0])
                storage.append(LumenHistFeatureExtractor.getInterquartileRangeFeatureValue()[0])
                storage.append(LumenHistFeatureExtractor.getRangeFeatureValue()[0])
                storage.append(LumenHistFeatureExtractor.getEntropyFeatureValue()[0])
                storage.append(LumenHistFeatureExtractor.getUniformityFeatureValue()[0])
                storage.append(LumenGLCMFeatureExtractor.getAutocorrelationFeatureValue()[0])
                storage.append(LumenGLCMFeatureExtractor.getClusterTendencyFeatureValue()[0])
                storage.append(LumenGLCMFeatureExtractor.getMaximumProbabilityFeatureValue()[0])
                storage.append(LumenGLCMFeatureExtractor.getContrastFeatureValue()[0])
                storage.append(LumenGLCMFeatureExtractor.getDifferenceEntropyFeatureValue()[0])
                storage.append(LumenGLCMFeatureExtractor.getDifferenceAverageFeatureValue()[0])
                storage.append(LumenGLCMFeatureExtractor.getJointEnergyFeatureValue()[0])
                storage.append(LumenGLCMFeatureExtractor.getJointEntropyFeatureValue()[0])
                storage.append(LumenGLCMFeatureExtractor.getIdFeatureValue()[0])
                storage.append(LumenGLCMFeatureExtractor.getImc1FeatureValue()[0])
                storage.append(LumenGLCMFeatureExtractor.getSumSquaresFeatureValue()[0])
                storage.append(LumenGLCMFeatureExtractor.getSumAverageFeatureValue()[0])
                storage.append(LumenGLCMFeatureExtractor.getSumEntropyFeatureValue()[0])
                storage.append(LumenGLCMFeatureExtractor.getClusterTendencyFeatureValue()[0])
                storage.append(LumenGLCMFeatureExtractor.getInverseVarianceFeatureValue()[0])
                storage.append(LumenGLCMFeatureExtractor.getIdmnFeatureValue()[0])
                storage.append(LumenGLSZMFeatureExtractor.getSmallAreaEmphasisFeatureValue()[0])
                storage.append(LumenGLSZMFeatureExtractor.getLargeAreaEmphasisFeatureValue()[0])
                storage.append(LumenGLSZMFeatureExtractor.getGrayLevelNonUniformityFeatureValue()[0])
                storage.append(LumenGLSZMFeatureExtractor.getGrayLevelNonUniformityNormalizedFeatureValue()[0])
                storage.append(LumenGLSZMFeatureExtractor.getSizeZoneNonUniformityFeatureValue()[0])
                storage.append(LumenGLSZMFeatureExtractor.getSizeZoneNonUniformityNormalizedFeatureValue()[0])
                storage.append(LumenGLSZMFeatureExtractor.getZonePercentageFeatureValue()[0])
                storage.append(LumenGLSZMFeatureExtractor.getGrayLevelVarianceFeatureValue()[0])
                storage.append(LumenGLSZMFeatureExtractor.getZoneVarianceFeatureValue()[0])
                storage.append(LumenGLSZMFeatureExtractor.getZoneEntropyFeatureValue()[0])
                storage.append(LumenGLSZMFeatureExtractor.getLowGrayLevelZoneEmphasisFeatureValue()[0])
                storage.append(LumenGLSZMFeatureExtractor.getHighGrayLevelZoneEmphasisFeatureValue()[0])
                storage.append(LumenGLSZMFeatureExtractor.getSmallAreaLowGrayLevelEmphasisFeatureValue()[0])
                storage.append(LumenGLSZMFeatureExtractor.getSmallAreaHighGrayLevelEmphasisFeatureValue()[0])
                storage.append(LumenGLSZMFeatureExtractor.getLargeAreaLowGrayLevelEmphasisFeatureValue()[0])
                storage.append(LumenGLSZMFeatureExtractor.getLargeAreaHighGrayLevelEmphasisFeatureValue()[0])

                # Outer
                storage.append(OuterHistFeatureExtractor.getMaximumFeatureValue()[0])
                storage.append(OuterHistFeatureExtractor.getMinimumFeatureValue()[0])
                storage.append(OuterHistFeatureExtractor.getMedianFeatureValue()[0])
                storage.append(OuterHistFeatureExtractor.getMeanFeatureValue()[0])
                storage.append(OuterHistFeatureExtractor.getVarianceFeatureValue()[0])
                storage.append(OuterHistFeatureExtractor.getEnergyFeatureValue()[0])
                storage.append(OuterHistFeatureExtractor.getStandardDeviationFeatureValue()[0])
                storage.append(OuterHistFeatureExtractor.getSkewnessFeatureValue()[0])
                storage.append(OuterHistFeatureExtractor.getKurtosisFeatureValue()[0])
                storage.append(OuterHistFeatureExtractor.getRootMeanSquaredFeatureValue()[0])
                storage.append(OuterHistFeatureExtractor.getInterquartileRangeFeatureValue()[0])
                storage.append(OuterHistFeatureExtractor.getRangeFeatureValue()[0])
                storage.append(OuterHistFeatureExtractor.getEntropyFeatureValue()[0])
                storage.append(OuterHistFeatureExtractor.getUniformityFeatureValue()[0])
                storage.append(OuterGLCMFeatureExtractor.getAutocorrelationFeatureValue()[0])
                storage.append(OuterGLCMFeatureExtractor.getClusterTendencyFeatureValue()[0])
                storage.append(OuterGLCMFeatureExtractor.getMaximumProbabilityFeatureValue()[0])
                storage.append(OuterGLCMFeatureExtractor.getContrastFeatureValue()[0])
                storage.append(OuterGLCMFeatureExtractor.getDifferenceEntropyFeatureValue()[0])
                storage.append(OuterGLCMFeatureExtractor.getDifferenceAverageFeatureValue()[0])
                storage.append(OuterGLCMFeatureExtractor.getJointEnergyFeatureValue()[0])
                storage.append(OuterGLCMFeatureExtractor.getJointEntropyFeatureValue()[0])
                storage.append(OuterGLCMFeatureExtractor.getIdFeatureValue()[0])
                storage.append(OuterGLCMFeatureExtractor.getImc1FeatureValue()[0])
                storage.append(OuterGLCMFeatureExtractor.getSumSquaresFeatureValue()[0])
                storage.append(OuterGLCMFeatureExtractor.getSumAverageFeatureValue()[0])
                storage.append(OuterGLCMFeatureExtractor.getSumEntropyFeatureValue()[0])
                storage.append(OuterGLCMFeatureExtractor.getClusterTendencyFeatureValue()[0])
                storage.append(OuterGLCMFeatureExtractor.getInverseVarianceFeatureValue()[0])
                storage.append(OuterGLCMFeatureExtractor.getIdmnFeatureValue()[0])
                storage.append(OuterGLSZMFeatureExtractor.getSmallAreaEmphasisFeatureValue()[0])
                storage.append(OuterGLSZMFeatureExtractor.getLargeAreaEmphasisFeatureValue()[0])
                storage.append(OuterGLSZMFeatureExtractor.getGrayLevelNonUniformityFeatureValue()[0])
                storage.append(OuterGLSZMFeatureExtractor.getGrayLevelNonUniformityNormalizedFeatureValue()[0])
                storage.append(OuterGLSZMFeatureExtractor.getSizeZoneNonUniformityFeatureValue()[0])
                storage.append(OuterGLSZMFeatureExtractor.getSizeZoneNonUniformityNormalizedFeatureValue()[0])
                storage.append(OuterGLSZMFeatureExtractor.getZonePercentageFeatureValue()[0])
                storage.append(OuterGLSZMFeatureExtractor.getGrayLevelVarianceFeatureValue()[0])
                storage.append(OuterGLSZMFeatureExtractor.getZoneVarianceFeatureValue()[0])
                storage.append(OuterGLSZMFeatureExtractor.getZoneEntropyFeatureValue()[0])
                storage.append(OuterGLSZMFeatureExtractor.getLowGrayLevelZoneEmphasisFeatureValue()[0])
                storage.append(OuterGLSZMFeatureExtractor.getHighGrayLevelZoneEmphasisFeatureValue()[0])
                storage.append(OuterGLSZMFeatureExtractor.getSmallAreaLowGrayLevelEmphasisFeatureValue()[0])
                storage.append(OuterGLSZMFeatureExtractor.getSmallAreaHighGrayLevelEmphasisFeatureValue()[0])
                storage.append(OuterGLSZMFeatureExtractor.getLargeAreaLowGrayLevelEmphasisFeatureValue()[0])
                storage.append(OuterGLSZMFeatureExtractor.getLargeAreaHighGrayLevelEmphasisFeatureValue()[0])

                # Vessel
                storage.append(VesselHistFeatureExtractor.getMaximumFeatureValue()[0])
                storage.append(VesselHistFeatureExtractor.getMinimumFeatureValue()[0])
                storage.append(VesselHistFeatureExtractor.getMedianFeatureValue()[0])
                storage.append(VesselHistFeatureExtractor.getMeanFeatureValue()[0])
                storage.append(VesselHistFeatureExtractor.getVarianceFeatureValue()[0])
                storage.append(VesselHistFeatureExtractor.getEnergyFeatureValue()[0])
                storage.append(VesselHistFeatureExtractor.getStandardDeviationFeatureValue()[0])
                storage.append(VesselHistFeatureExtractor.getSkewnessFeatureValue()[0])
                storage.append(VesselHistFeatureExtractor.getKurtosisFeatureValue()[0])
                storage.append(VesselHistFeatureExtractor.getRootMeanSquaredFeatureValue()[0])
                storage.append(VesselHistFeatureExtractor.getInterquartileRangeFeatureValue()[0])
                storage.append(VesselHistFeatureExtractor.getRangeFeatureValue()[0])
                storage.append(VesselHistFeatureExtractor.getEntropyFeatureValue()[0])
                storage.append(VesselHistFeatureExtractor.getUniformityFeatureValue()[0])
                storage.append(VesselGLCMFeatureExtractor.getAutocorrelationFeatureValue()[0])
                storage.append(VesselGLCMFeatureExtractor.getClusterTendencyFeatureValue()[0])
                storage.append(VesselGLCMFeatureExtractor.getMaximumProbabilityFeatureValue()[0])
                storage.append(VesselGLCMFeatureExtractor.getContrastFeatureValue()[0])
                storage.append(VesselGLCMFeatureExtractor.getDifferenceEntropyFeatureValue()[0])
                storage.append(VesselGLCMFeatureExtractor.getDifferenceAverageFeatureValue()[0])
                storage.append(VesselGLCMFeatureExtractor.getJointEnergyFeatureValue()[0])
                storage.append(VesselGLCMFeatureExtractor.getJointEntropyFeatureValue()[0])
                storage.append(VesselGLCMFeatureExtractor.getIdFeatureValue()[0])
                storage.append(VesselGLCMFeatureExtractor.getImc1FeatureValue()[0])
                storage.append(VesselGLCMFeatureExtractor.getSumSquaresFeatureValue()[0])
                storage.append(VesselGLCMFeatureExtractor.getSumAverageFeatureValue()[0])
                storage.append(VesselGLCMFeatureExtractor.getSumEntropyFeatureValue()[0])
                storage.append(VesselGLCMFeatureExtractor.getClusterTendencyFeatureValue()[0])
                storage.append(VesselGLCMFeatureExtractor.getInverseVarianceFeatureValue()[0])
                storage.append(VesselGLCMFeatureExtractor.getIdmnFeatureValue()[0])
                storage.append(VesselGLSZMFeatureExtractor.getSmallAreaEmphasisFeatureValue()[0])
                storage.append(VesselGLSZMFeatureExtractor.getLargeAreaEmphasisFeatureValue()[0])
                storage.append(VesselGLSZMFeatureExtractor.getGrayLevelNonUniformityFeatureValue()[0])
                storage.append(VesselGLSZMFeatureExtractor.getGrayLevelNonUniformityNormalizedFeatureValue()[0])
                storage.append(VesselGLSZMFeatureExtractor.getSizeZoneNonUniformityFeatureValue()[0])
                storage.append(VesselGLSZMFeatureExtractor.getSizeZoneNonUniformityNormalizedFeatureValue()[0])
                storage.append(VesselGLSZMFeatureExtractor.getZonePercentageFeatureValue()[0])
                storage.append(VesselGLSZMFeatureExtractor.getGrayLevelVarianceFeatureValue()[0])
                storage.append(VesselGLSZMFeatureExtractor.getZoneVarianceFeatureValue()[0])
                storage.append(VesselGLSZMFeatureExtractor.getZoneEntropyFeatureValue()[0])
                storage.append(VesselGLSZMFeatureExtractor.getLowGrayLevelZoneEmphasisFeatureValue()[0])
                storage.append(VesselGLSZMFeatureExtractor.getHighGrayLevelZoneEmphasisFeatureValue()[0])
                storage.append(VesselGLSZMFeatureExtractor.getSmallAreaLowGrayLevelEmphasisFeatureValue()[0])
                storage.append(VesselGLSZMFeatureExtractor.getSmallAreaHighGrayLevelEmphasisFeatureValue()[0])
                storage.append(VesselGLSZMFeatureExtractor.getLargeAreaLowGrayLevelEmphasisFeatureValue()[0])
                storage.append(VesselGLSZMFeatureExtractor.getLargeAreaHighGrayLevelEmphasisFeatureValue()[0])
                
                print(p, a, s, ",,,", len(storage))
                
                if len(storage) != 172:
                    exceptslice.append(str(p)+str(a)+str(s))
                
                FeatureStorage.append(storage)

    data = pd.DataFrame(FeatureStorage)

    if args.mode == 'train':
        data.to_csv('train_radiomics.csv', index=False)
    else:
        data.to_csv('test_radiomics.csv', index=False)

if __name__ == "__main__":
    main()
