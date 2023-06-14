_base_ = [
    '/opt/ml/level2_cv_semanticsegmentation-cv-01/baseline/mmsegmentation/configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py'
]
model = dict(
    backbone=dict(in_channels=1),
    decode_head=dict(num_classes=29)
)
