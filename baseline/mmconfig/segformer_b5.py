_base_ = [
    '/opt/ml/level2_cv_semanticsegmentation-cv-01/baseline/mmsegmentation/configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-640x640.py'
]
model = dict(
    decode_head=dict(num_classes=29),
)
