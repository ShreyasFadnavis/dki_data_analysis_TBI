import nibabel as nib
import numpy as np
import dipy.reconst.dki as dki
import dipy.reconst.dti as dti
from dipy.segment.mask import median_otsu
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
from dipy.io.image import save_nifti
import dipy.reconst.fwdti as fwdti
import dipy.reconst.msdki as msdki


fdwi = "/home/shfadn/Data/TBI/Subj24/aTBIp2_024_00m_jj.ready_dipy/aTBIp2_024_00m_jj/DKI/3shells/eddy_corrected_96dir.nii.gz"
fbval = "/home/shfadn/Data/TBI/Subj24/aTBIp2_024_00m_jj.ready_dipy/aTBIp2_024_00m_jj/DKI/3shells/96dir.bval"
fbvec = "/home/shfadn/Data/TBI/Subj24/aTBIp2_024_00m_jj.ready_dipy/aTBIp2_024_00m_jj/DKI/3shells/eddy_corrected_96dir.bvec"

img = nib.load(fdwi)
data = img.get_data()
affine = img.affine

maskdata, mask = median_otsu(data, vol_idx=[0, 1], dilate=1)

bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

include = np.ones(97)
include[33] = 0
include[66] = 0
bvals = bvals[include > 0.1]
bvecs = bvecs[include > 0.1]
data = data[:, :, :, include > 0.1]

gtab = gradient_table(bvals, bvecs)

dkimodel = dki.DiffusionKurtosisModel(gtab)

dkifit = dkimodel.fit(data, mask=mask)

FA = dkifit.fa
save_nifti('sub24_00_DKI_FA.nii.gz', FA, affine)
MD = dkifit.md
save_nifti('sub24_00_DKI_MD.nii.gz', MD, affine)
AD = dkifit.ad
save_nifti('sub24_00_DKI_AD.nii.gz', AD, affine)
RD = dkifit.rd
save_nifti('sub24_00_DKI_RD.nii.gz', RD, affine)

tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(data, mask=mask)

dti_FA = tenfit.fa
save_nifti('sub24_00_DTI_FA.nii.gz', dti_FA, affine)
dti_MD = tenfit.md
save_nifti('sub24_00_DTI_MD.nii.gz', dti_MD, affine)
dti_AD = tenfit.ad
save_nifti('sub24_00_DTI_AD.nii.gz', dti_AD, affine)
dti_RD = tenfit.rd
save_nifti('sub24_00_DTI_RD.nii.gz', dti_RD, affine)

###############################################################################
# MSDKI
msdki_model = msdki.MeanDiffusionKurtosisModel(gtab)
msdki_fit = msdki_model.fit(data, mask=mask)
MSK = msdki_fit.msk
save_nifti('sub24_00_DKI_MSK.nii.gz', MSK, affine)
###############################################################################
# FW-DTI

fwdtimodel = fwdti.FreeWaterTensorModel(gtab)
fwdtifit = fwdtimodel.fit(data, mask=mask)

FW_FA = fwdtifit.fa
save_nifti('sub24_00_DTI_FW_FA.nii.gz', FW_FA, affine)
FW_MD = fwdtifit.md
save_nifti('sub24_00_DTI_FW_MD.nii.gz', FW_MD, affine)
