_base_ = ['/opt/ml/level2_cv_semanticsegmentation-cv-01/baseline/mmsegmentation/configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py']

data_preprocessor=dict(
    mean=[0.121,0.121,0.121],
    std=[0.1641,0.1641,0.1641],
    bhr_to_rgb = False
)
model = dict(
    decode_head=dict(num_classes = 29)
)