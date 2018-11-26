import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal
from LSTM.BiConvLSTM import BiConvLSTM



def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.05,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.05,inplace=True)
        )

def conv_no_lrelu(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
        )
def predict_image(in_planes):
    return nn.Conv2d(in_planes,1,kernel_size=3,stride=1,padding=1,bias=False)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.05,inplace=True)
    )


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]


class Gen_Guided_UNet(nn.Module):
    expansion = 1

    def __init__(self,batchNorm=True,input_size=[240,416],is_training=True):
        super(Gen_Guided_UNet,self).__init__()
        self.batchNorm = batchNorm
        self.is_training = is_training
        self.pre_conv1   = conv(self.batchNorm, 1, 64,  kernel_size=3,  stride=1)
        self.pre_conv1_1 = conv(self.batchNorm, 64, 64, kernel_size=3,  stride=1)

        self.pre_conv2   = conv(self.batchNorm, 1, 64,  kernel_size=3,  stride=1)
        self.pre_conv2_1 = conv(self.batchNorm, 64, 64, kernel_size=3,  stride=1)

        self.pre_conv3   = conv(self.batchNorm, 1, 64,  kernel_size=3,  stride=1)
        self.pre_conv3_1 = conv(self.batchNorm, 64, 64, kernel_size=3,  stride=1)

        self.biconvlstm  = BiConvLSTM(input_size=(input_size[0], input_size[1]), input_dim=64, hidden_dim=64,kernel_size=(3, 3), num_layers=1)

        self.LSTM_out = conv(self.batchNorm,128,64,  kernel_size=1,  stride=1)

        self.conv1_mask = conv(self.batchNorm, 1, 64,  kernel_size=3,  stride=1)
        self.conv2_mask = conv(self.batchNorm, 64, 64, kernel_size=3, stride=1)

        self.conv1   = conv(self.batchNorm,   64, 128,  kernel_size=7,  stride=2)#64
        self.conv1_1 = conv(self.batchNorm,   128,128)  # 128*128 ->64*64
        self.conv2   = conv(self.batchNorm,   128,256,  kernel_size=3,  stride=2)#64 ->32
        self.conv2_1 = conv(self.batchNorm,   256,256)  # 128*128 ->64*64
        self.conv3   = conv(self.batchNorm,   256,512,  kernel_size=3,  stride=2)#32->16
        self.conv3_1 = conv(self.batchNorm,   512,512)
        self.conv4   = conv(self.batchNorm,   512,1024, kernel_size=3,  stride=2)#16->8
        self.conv4_1 = conv(self.batchNorm,   1024,1024)


        self.deconv4 = deconv(1024,512)
        self.deconv3 = deconv(1025,256)
        self.deconv2 = deconv(513,128)
        self.deconv1 = deconv(257,64)

        self.predict_image4 = predict_image(1024)
        self.predict_image3 = predict_image(1025)
        self.predict_image2 = predict_image(513)
        self.predict_image1 = predict_image(257)


        self.upsampled_image4_to_3 = nn.ConvTranspose2d(1,1, 4, 2, 1, bias=False)#8_16
        self.upsampled_image3_to_2 = nn.ConvTranspose2d(1,1, 4, 2, 1, bias=False)#16-32
        self.upsampled_image2_to_1 = nn.ConvTranspose2d(1,1, 4, 2, 1, bias=False)#32-64
        self.upsampled_image1_to_finally = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)  # 64-128

        self.output1 = conv(self.batchNorm,129,64,kernel_size=3,stride=1)
        self.output2 = conv(self.batchNorm, 64, 64, kernel_size=3, stride=1)
        self.output3 = conv_no_lrelu(self.batchNorm,64,1,kernel_size=3,stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data,a=0.05)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, data1,data2,data3,mask):

        CNN_seq = []
        pre_conv1 = self.pre_conv1(data1)
        pre_conv1_1 = self.pre_conv1_1(pre_conv1)
        CNN_seq.append(pre_conv1_1)

        pre_conv2 = self.pre_conv2(data2)
        pre_conv2_1 = self.pre_conv2_1(pre_conv2)
        CNN_seq.append(pre_conv2_1)

        pre_conv3 = self.pre_conv3(data3)
        pre_conv3_1 = self.pre_conv3_1(pre_conv3)
        CNN_seq.append(pre_conv3_1)

        CNN_seq_out      = torch.stack(CNN_seq, dim=1)
        CNN_seq_feature_maps = self.biconvlstm(CNN_seq_out)
        # CNN_concat_input = CNN_seq_out[:, 1, ...]+CNN_seq_feature_maps[:, 1, ...]
        CNN_concat_input = torch.cat([CNN_seq_out[:, 1, ...],CNN_seq_feature_maps[:, 1, ...]],dim=1)
       
        LSTM_out         = self.LSTM_out(CNN_concat_input)#128*128*64

        conv1_mask = self.conv1_mask(mask)
        conv2_mask = self.conv2_mask(conv1_mask)#128*128*64

        out_conv1 = self.conv1_1(self.conv1(LSTM_out))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))

        out_conv1_mask = self.conv1_1(self.conv1(conv2_mask))
        out_conv2_mask = self.conv2_1(self.conv2(out_conv1_mask))
        out_conv3_mask = self.conv3_1(self.conv3(out_conv2_mask))
        out_conv4_mask = self.conv4_1(self.conv4(out_conv3_mask))

        sum4 = out_conv4+out_conv4_mask
        image_4       = self.predict_image4(sum4)
        image_4_up    = crop_like(self.upsampled_image4_to_3(image_4), out_conv3)
        out_deconv3 = crop_like(self.deconv4(sum4), out_conv3)

        sum3 = out_conv3 + out_conv3_mask
        concat3 = torch.cat((sum3,out_deconv3,image_4_up),dim=1)
        image_3       = self.predict_image3(concat3)
        image_3_up    = crop_like(self.upsampled_image3_to_2(image_3), out_conv2)
        out_deconv2 = crop_like(self.deconv3(concat3), out_conv2)

        sum2 = out_conv2+out_conv2_mask
        concat2 = torch.cat((sum2,out_deconv2,image_3_up),dim=1)
        image_2       = self.predict_image2(concat2)
        image_2_up    = crop_like(self.upsampled_image2_to_1(image_2), out_conv1)
        out_deconv2 = crop_like(self.deconv2(concat2), out_conv1)

        sum1 = out_conv1 + out_conv1_mask
        concat1 = torch.cat((sum1,out_deconv2,image_2_up),dim=1)
        image_1 = self.predict_image1(concat1)
        image_1_up = crop_like(self.upsampled_image1_to_finally(image_1), LSTM_out)
        # print(image_1_up.shape)
        out_deconv1 = crop_like(self.deconv1(concat1), LSTM_out)

        sum0 = LSTM_out + conv2_mask
        concat0 = torch.cat([sum0,out_deconv1,image_1_up],dim=1)
        image_out       = self.output1(concat0)
        image_out2 = self.output2(image_out)
        image_finally = self.output3(image_out2)

        image_finally = torch.clamp(image_finally,0.,1.)
        # print('image_1',image_finally.shape)

        if self.is＿training:
            return image_4,image_3,image_2,image_1,image_finally
        else:
            return image_finally

    