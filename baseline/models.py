import torch.nn as nn

from torchvision import models
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders.mix_transformer import Mlp
import torch

from mmsegmentation.mmseg.apis import init_model
from mmengine import Config
def fcn_resnet50(class_len):
    model = models.segmentation.fcn_resnet50(pretrained=True)

    # output class를 data set에 맞도록 수정
    model.classifier[4] = nn.Conv2d(512, class_len, kernel_size=1)
    return model
def UNet(class_len):
    model = smp.Unet(encoder_name="resnet101",encoder_weights='imagenet',classes = class_len)
    return model

class ConvModule(nn.Module):
    def __init__(self,input_dim,output_dim,stride =1) :
        super().__init__()
        self.conv = nn.Conv2d(input_dim,output_dim,stride)
        self.norm = nn.SyncBatchNorm(output_dim)
        self.act = nn.GELU()
        nn.init.kaiming_normal(self.conv.weight)
    def forward(self,input):
        output = self.act(self.norm(self.conv(input)))
        return output
    
class SegFormerDecoder(nn.Module):
    def __init__(self, input_dim=[32,64,160,256],feature_stride=[4,8,16,32],channels=128,dropout_ratio=0.1,num_classes=29,embedding_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.feature_stride = feature_stride
        self.channels = channels
        self.dropout_ratio=dropout_ratio
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        

        i1,i2,i3,i4 = input_dim
        self.dropout = nn.Dropout(dropout_ratio)
        # self.mlp1 = nn.Conv2d(in_channels=i1,out_channels=embedding_dim,kernel_size=1)
        self.mlp1 = ConvModule(i1,embedding_dim)
       # self.up1 = nn.Upsample(scale_factor=4,mode='bilinear')
        # self.mlp2 = nn.Conv2d(in_channels=i2,out_channels=embedding_dim,kernel_size=1)
        self.mlp2 = ConvModule(i2,embedding_dim)
        self.up2 = nn.Upsample(scale_factor=2,mode='bilinear')
        # self.mlp3 = nn.Conv2d(in_channels=i3,out_channels=embedding_dim,kernel_size=1)
        self.mlp3 = ConvModule(i3,embedding_dim)
        self.up3 = nn.Upsample(scale_factor=4,mode='bilinear')
        # self.mlp4 = nn.Conv2d(in_channels=i4,out_channels=embedding_dim,kernel_size=1)
        self.mlp4 = ConvModule(i4,embedding_dim)
        self.up4 = nn.Upsample(scale_factor=8,mode='bilinear')

        # self.fuse_mlp = nn.Conv2d(in_channels= embedding_dim*4,out_channels=embedding_dim,kernel_size=1)
        self.fuse_mlp = ConvModule(embedding_dim*4,embedding_dim)
        self.seg_pred = nn.Conv2d(embedding_dim,num_classes,1)
        self.up_to_4 = nn.Upsample(scale_factor=4)
    def forward(self,input):
        c1,c2,c3,c4 = input[2:]
        c1 = self.mlp1(c1)
        #c1 = self.up1(c1)
        c2 = self.mlp2(c2)
        c2 = self.up2(c2)
        c3 = self.mlp3(c3)
        c3 = self.up3(c3)
        c4 = self.mlp4(c4)
        c4 = self.up4(c4)
        c = torch.concat((c1,c2,c3,c4),dim=1)
        c = self.fuse_mlp(c)
        c = self.dropout(c)
        output = self.seg_pred(c)
        output = self.up_to_4(output)
        return output

class SegFormer(nn.Module):
    def __init__(self, class_num=29):
        super().__init__()
        self.encoder = smp.encoders.get_encoder('mit_b0',weights='imagenet')
        self.decoder = SegFormerDecoder()

    def forward(self,input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return decoded

def GetSegFormer(class_len):
    model = SegFormer(29)
    return model

class MMSegFormer():
    def __init__(self) -> None:
        cfg=Config.fromfile('/opt/ml/level2_cv_semanticsegmentation-cv-01/baseline/mmconfig/segformer.py')
        checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_512x512_160k_ade20k/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth'
        self.model = init_model(cfg,checkpoint)
        self.upsample = nn.Upsample(scale_factor=2)
    def forward(self,input):
        output = self.model(input)
        output = self.upsample(output)
        return output

