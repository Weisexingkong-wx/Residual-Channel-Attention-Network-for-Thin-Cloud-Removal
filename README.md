# Residual-Channel-Attention-Network-for-Thin-Cloud-Removal
This repository is an implementation of "An Effective Network Integrating Residual Learning and Channel Attention Mechanism for Thin Cloud Removal", in IEEE GEOSCIENCE AND REMOTE SENSING LETTERS 2022.

# Requirements
-Python3 (tested with 3.6)

-Tensorflow (tested with 1.9)

-cuda (tested with 9.0)

-cudnn (tested with 7.5)

-OpenCV

-tflearn

-matplotlib

# Training examples
python CloudRemoval_net_train.py

# Testing examples
python test.py

# Evaluating results
There are two types of evaluation indexs, including indexs relying on reference images showing in calculate_PSNR_SSIM.py and indexs without relying on reference images showing in .m files, where Entropy.m calculates image entropy (IE)  and Laplace.m calculates Laplace gradient (LG).

# License
Academic use only.
