_base_ = ['/opt/ml/level2_cv_semanticsegmentation-cv-01/baseline/mmsegmentation/configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-512x512.py']

# model settings
model = dict(
    pretrained='pretrain/mit_b5.pth',
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512],num_classes=29))