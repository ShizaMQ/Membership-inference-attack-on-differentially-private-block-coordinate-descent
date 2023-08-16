# Membership Inference Attack (MIA)

This repository uses MIA code mainly from https://github.com/tensorflow/privacy/tree/master/tensorflow_privacy/privacy/privacy_tests with some amendments.
It implements MIA on DPDL model trained on our Differentially Private version of Block Coordinate Descent (DP-BCD) Algorithm and evaluate its robustness againstan adversary trying to infer sensitive information about the training dataset of the model.