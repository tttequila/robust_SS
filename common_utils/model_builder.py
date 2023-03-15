import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from argparse import Namespace
from av_hubert import fairseq
from fairseq import checkpoint_utils, options, tasks, utils

# builder for facial attributes analysis stream
def facial_net(pool_type='maxpool', input_channel=3, fc_out=512, with_fc=False, weights=''):
    pretrained = False
    original_resnet = torchvision.models.resnet18(pretrained)
    net = Resnet18(original_resnet, pool_type=pool_type, with_fc=with_fc, fc_in=512, fc_out=fc_out)

    if len(weights) > 0:
        print('Loading weights for facial attributes analysis stream')
        pretrained_state = torch.load(weights)
        model_state = net.state_dict()
        pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
        model_state.update(pretrained_state)
        net.load_state_dict(model_state)
    return net

def lip_reading_net(weights=''):
    net = VisualFrontend()

    if len(weights) > 0:
        print('Loading weights for visual frontend')
        pretrained_state = torch.load(weights)
        model_state = net.state_dict()
        pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
        model_state.update(pretrained_state)
        net.load_state_dict(model_state)
    return net

def av_hubert_net(user_dir, weights):
    '''
        user_dir: model stucture
        weights: model weights
    '''
    print('Loading weights for AV_HuBERT')
    utils.import_user_module(Namespace(user_dir=user_dir))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([weights])
    model = models[0]
    return model

def av_hubert_feature(model, video, audio=None, output_layer=None):
    '''return the feature given by aHuBERT'''
    
    # B, num_frame, Feature_dim
    feature, _ = model.extract_finetune(source={'video':video, 'audio':audio}, padding_mask=None, output_layer=output_layer)
    
    return feature

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

class Resnet18(nn.Module):
    def __init__(self, original_resnet, pool_type='maxpool', input_channel=3, with_fc=False, fc_in=512, fc_out=512):
        super(Resnet18, self).__init__()
        self.pool_type = pool_type
        self.input_channel = input_channel
        self.with_fc = with_fc

        #customize first convolution layer to handle different number of channels for images and spectrograms
        self.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        layers = [self.conv1]
        layers.extend(list(original_resnet.children())[1:-2])
        self.feature_extraction = nn.Sequential(*layers) #features before pooling

        if with_fc:
            self.fc = nn.Linear(fc_in, fc_out)
            self.fc.apply(weights_init)

    def forward(self, x):
        x = self.feature_extraction(x)

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)
        else:
            return x

        if self.with_fc:
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x.view(x.size(0), -1)
        else:
            return x.view(x.size(0), -1)

    def forward_multiframe(self, x, pool=True):
        (B, T, C, H, W) = x.size()
        x = x.contiguous()
        x = x.view(B * T, C, H, W)
        x = self.feature_extraction(x)

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        if not pool:
            return x 

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool3d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool3d(x, 1)
        
        if self.with_fc:
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x.view(x.size(0), -1, 1, 1)
        else:
            return x.view(x.size(0), -1, 1, 1)
        return x

class ResNetLayer(nn.Module):

    """
    A ResNet layer used to build the ResNet network.
    Architecture:
    --> conv-bn-relu -> conv -> + -> bn-relu -> conv-bn-relu -> conv -> + -> bn-relu -->
     |                        |   |                                    |
     -----> downsample ------>    ------------------------------------->
    """

    def __init__(self, inplanes, outplanes, stride):
        super(ResNetLayer, self).__init__()
        self.conv1a = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=(1,1), stride=stride, bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        return


    def forward(self, inputBatch):
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        batch = self.conv2a(batch)
        if self.stride == 1:
            residualBatch = inputBatch
        else:
            residualBatch = self.downsample(inputBatch)
        batch = batch + residualBatch
        intermediateBatch = batch
        batch = F.relu(self.outbna(batch))

        batch = F.relu(self.bn1b(self.conv1b(batch)))
        batch = self.conv2b(batch)
        residualBatch = intermediateBatch
        batch = batch + residualBatch
        outputBatch = F.relu(self.outbnb(batch))
        return outputBatch

class ResNet(nn.Module):

    """
    An 18-layer ResNet architecture.
    """

    def __init__(self):
        super(ResNet, self).__init__()
        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 128, stride=2)
        self.layer3 = ResNetLayer(128, 256, stride=2)
        self.layer4 = ResNetLayer(256, 256, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(4,4), stride=(1,1))
        return


    def forward(self, inputBatch):
        batch = self.layer1(inputBatch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        outputBatch = self.avgpool(batch)
        return outputBatch

class VisualFrontend(nn.Module):

    """
    A visual feature extraction module. Generates a 256-dim feature vector per video frame.
    Architecture: A 3D convolution block followed by an 18-layer ResNet.
    """

    def __init__(self, causal=False):
        super(VisualFrontend, self).__init__()
        self.causal = causal

        if self.causal:
            self.conv = nn.Conv3d(1, 64, kernel_size=(5,7,7), stride=(1,2,2), padding=(4,3,3), bias=False)
        else:
            self.conv = nn.Conv3d(1, 64, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False)
        self.norm = nn.BatchNorm3d(64, momentum=0.01, eps=0.001)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

        self.resnet = ResNet()
        return


    def forward(self, inputBatch):
        inputBatch = inputBatch.unsqueeze(1)
        batchsize = inputBatch.shape[0]
        batch = self.conv(inputBatch)
        if self.causal:
            batch = batch[:,:,:-4,:,:]
        batch = self.norm(batch)
        batch = self.act(batch)
        batch = self.pool(batch)

        batch = batch.transpose(1, 2)
        batch = batch.reshape(batch.shape[0]*batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4])
        outputBatch = self.resnet(batch)
        outputBatch = outputBatch.reshape(batchsize, -1, 256)
        return outputBatch

class videoEncoder(nn.Module):
    def __init__(self,B=256, H=512, R=5, causal=False):
        super(videoEncoder, self).__init__()
        ve_blocks = []
        for x in range(R):
            ve_blocks +=[VisualConv1D(B,H, causal=causal)]
        self.net = nn.Sequential(*ve_blocks)
        self.conv1x1 = nn.Conv1d(B, 256, 1)

    def forward(self, v):
        v = v.transpose(1,2)
        v = self.net(v)
        v = self.conv1x1(v)
        return v
    
class AVHuBERT_Adapter(nn.Module):
    '''turn the extracted feature from av_hubert to 256 for Separator input'''
    def __init__(self,B=768, H=256, Num_Layers=1):
        super(AVHuBERT_Adapter, self).__init__()
        self.proj = nn.Linear(B*Num_Layers, H, bias=True)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.transpose(1, 2)
        return x

class VisualConv1D(nn.Module):
    def __init__(self, B=256, H=512, kernel_size=3, dilation=1, causal = False):
        super(VisualConv1D, self).__init__()
        self.relu_0 = nn.ReLU()
        self.norm_0 = nn.BatchNorm1d(B)
        self.conv1x1 = nn.Conv1d(B, H, 1)
        self.relu = nn.ReLU()
        self.norm_1 = nn.BatchNorm1d(H)
        self.dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))

        self.dsconv = nn.Conv1d(H, H, kernel_size, stride=1, padding=self.dconv_pad,dilation=1, groups=H)
        self.prelu = nn.PReLU()
        self.norm_2 = nn.BatchNorm1d(H)
        self.pw_conv = nn.Conv1d(H, B, 1)
        self.causal = causal

    def forward(self, x):
        out = self.relu_0(x)
        out = self.norm_0(out)
        out = self.conv1x1(out)
        out = self.relu(out)
        out = self.norm_1(out)
        out = self.dsconv(out)
        if self.causal:
            out = out[:, :, :-self.dconv_pad]
        out = self.prelu(out)
        out = self.norm_2(out)
        out = self.pw_conv(out)
        return out + x

class Dual_RNN_Block(nn.Module):
    def __init__(self, out_channels,
                 hidden_channels, rnn_type='LSTM',
                 dropout=0, bidirectional=False, num_spks=2):
        super(Dual_RNN_Block, self).__init__()
        # RNN model
        self.intra_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.inter_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        # Norm
        self.intra_norm = nn.GroupNorm(1, out_channels, eps=1e-8)
        self.inter_norm = nn.GroupNorm(1, out_channels, eps=1e-8)
        # Linear
        self.intra_linear = nn.Linear(
            hidden_channels*2 if bidirectional else hidden_channels, out_channels)
        self.inter_linear = nn.Linear(
            hidden_channels*2 if bidirectional else hidden_channels, out_channels)
        
    def forward(self, x):
        '''
           x: [B, N, K, S]
           out: [Spks, B, N, K, S]
        '''
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B*S, K, N)
        # [BS, K, H]
        intra_rnn, _ = self.intra_rnn(intra_rnn)
        # [BS, K, N]
        intra_rnn = self.intra_linear(intra_rnn.contiguous().view(B*S*K, -1)).view(B*S, K, -1)
        # [B, S, K, N]
        intra_rnn = intra_rnn.view(B, S, K, N)
        # [B, N, K, S]
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)
        
        # [B, N, K, S]
        intra_rnn = intra_rnn + x

        # inter RNN
        # [BK, S, N]
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B*K, S, N)
        # [BK, S, H]
        inter_rnn, _ = self.inter_rnn(inter_rnn)
        # [BK, S, N]
        inter_rnn = self.inter_linear(inter_rnn.contiguous().view(B*S*K, -1)).view(B*K, S, -1)
        # [B, K, S, N]
        inter_rnn = inter_rnn.view(B, K, S, N)
        # [B, N, K, S]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, N, K, S]
        out = inter_rnn + intra_rnn

        return out

class Separator_DPRNN(nn.Module):
    def __init__(self, enc_dim=512, fea_dim=128, hidden_dim=128, segment_size=100, R_a=3, R_av=3):
        super(Separator_DPRNN, self).__init__()
        self.K , self.R_a, self.R_av = segment_size, R_a, R_av
        self.layer_norm = nn.GroupNorm(1, enc_dim, eps=1e-8)
        self.bottleneck_conv1x1 = nn.Conv1d(enc_dim, fea_dim, 1, bias=False)
        # self.av_conv = nn.Conv1d(fea_dim*2, fea_dim, 1, bias=False)
        
        self.dual_rnn_a = nn.ModuleList([])
        for i in range(R_a):
            self.dual_rnn_a.append(Dual_RNN_Block(fea_dim, hidden_dim,
                                     rnn_type='LSTM',  dropout=0,
                                     bidirectional=True))
        self.dual_rnn_av = nn.ModuleList([])
        for i in range(R_av):
            self.dual_rnn_av.append(Dual_RNN_Block(2*fea_dim, 3*hidden_dim,
                                     rnn_type='LSTM',  dropout=0,
                                     bidirectional=True))

        self.mask = nn.Sequential(nn.Conv1d(2*fea_dim, enc_dim, 1, bias=False),
                                  nn.ReLU(inplace=True))

    def forward(self, aud, visual):
        B, N, L = aud.size()
        aud = self.layer_norm(aud) # [B, N, L]
        aud = self.bottleneck_conv1x1(aud) # [B, F, L]

        aud, gap = self._Segmentation(aud, self.K) # [B, F, k, S]
        for i in range(self.R_a):
            aud = self.dual_rnn_a[i](aud)
        aud = self._over_add(aud, gap)

        # Audio-visual fusion
        fusion = torch.cat((aud, visual), 1)  # [B, 2*F, L]
        # x  = self.av_conv(x)

        fusion, gap = self._Segmentation(fusion, self.K) # [B, 2*F, k, S]
        for i in range(self.R_av):
            fusion = self.dual_rnn_av[i](fusion)
        fusion = self._over_add(fusion, gap)

        mask = self.mask(fusion)

        return mask

    def _padding(self, input, K):
        '''
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        '''
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        '''
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        '''
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(
            B, N, -1, K).transpose(2, 3)

        return input.contiguous(), gap


    def _over_add(self, input, gap):
        '''
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        '''
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input

class Separator_blind_source(nn.Module):
    def __init__(self, enc_dim=512, fea_dim=128, hidden_dim=128, segment_size=100, R_a=6):
        super(Separator_blind_source, self).__init__()
        self.K , self.R_a = segment_size, R_a
        self.layer_norm = nn.GroupNorm(1, enc_dim, eps=1e-8)
        self.bottleneck_conv1x1 = nn.Conv1d(enc_dim, fea_dim, 1, bias=False)
        self.av_conv = nn.Conv1d(fea_dim*2, fea_dim, 1, bias=False)
        
        self.dual_rnn_a = nn.ModuleList([])
        for i in range(R_a):
            self.dual_rnn_a.append(Dual_RNN_Block(fea_dim, hidden_dim,
                                     rnn_type='LSTM',  dropout=0,
                                     bidirectional=True))
        self.dual_rnn_av = nn.ModuleList([])

        self.prelu = nn.PReLU()
        self.mask_conv1x1 = nn.Conv1d(fea_dim, 2*enc_dim, 1, bias=False)

    def forward(self, x):
        B, N, L = x.size()
        x = self.layer_norm(x) # [B, N, L]
        x = self.bottleneck_conv1x1(x) # [B, F, L]

        x, gap = self._Segmentation(x, self.K) # [B, F, k, S]
        for i in range(self.R_a):
            x = self.dual_rnn_a[i](x)
        x = self._over_add(x, gap)

        x = self.prelu(x)
        x = self.mask_conv1x1(x)

        return x

    def _padding(self, input, K):
        '''
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        '''
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        '''
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        '''
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(
            B, N, -1, K).transpose(2, 3)

        return input.contiguous(), gap


    def _over_add(self, input, gap):
        '''
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        '''
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input

class DepthConv1d(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True, causal=False):
        super(DepthConv1d, self).__init__()
        
        self.causal = causal
        self.skip = skip
        
        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding
        self.dconv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel, dilation=dilation,
          groups=hidden_channel,
          padding=self.padding)
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        if self.causal:
            self.reg1 = cLN(hidden_channel, eps=1e-08)
            self.reg2 = cLN(hidden_channel, eps=1e-08)
        else:
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        
        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.conv1d(input)))
        if self.causal:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)[:,:,:-self.padding]))
        else:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual

class Separator_TCN(nn.Module):
    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim,
                 layer, stack_a, stack_av, kernel=3, skip=True, 
                 causal=False, dilated=True):
        super(Separator_TCN, self).__init__()
        
        # input is a sequence of features of shape (B, N, L)
        
        # normalization
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8)

        self.BN = nn.Conv1d(input_dim, BN_dim, 1)
        
        # TCN for feature extraction
        self.receptive_field = 0
        self.dilated = dilated
        
        self.TCN_a = nn.ModuleList([])
        for s in range(stack_a):
            for i in range(layer):
                if self.dilated:
                    self.TCN_a.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=2**i, padding=2**i, skip=skip, causal=causal)) 
                else:
                    self.TCN_a.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal))   
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2**i
                    else:
                        self.receptive_field += (kernel - 1)

        self.TCN_av = nn.ModuleList([])
        for s in range(stack_av):
            for i in range(layer):
                if self.dilated:
                    self.TCN_av.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=2**i, padding=2**i, skip=skip, causal=causal)) 
                else:
                    self.TCN_av.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal))   
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2**i
                    else:
                        self.receptive_field += (kernel - 1)       
        #print("Receptive field: {:3d} frames.".format(self.receptive_field))
        
        self.conv_proj = nn.Conv1d(BN_dim*2, BN_dim, 1, bias=False)
        # output layer
        
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(BN_dim, output_dim, 1)
                                   )
        
        self.skip = skip
        
    def forward(self, input, ref):
        
        # input shape: (B, N, L)
       
        
        # normalization
        output = self.BN(self.LN(input))
        
        # pass to audio TCN
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN_a)):
                residual, skip = self.TCN_a[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN_a)):
                residual = self.TCN_a[i](output)
                output = output + residual

        # feature fusion
        if self.skip:
            output = torch.cat((skip_connection, ref), dim=1)
        else:
            output = torch.cat((output, ref), dim=1)
        
        output = self.conv_proj(output)
        
        # pass to av TCN
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN_av)):
                residual, skip = self.TCN_av[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN_av)):
                residual = self.TCN_av[i](output)
                output = output + residual
            
        # output layer
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)
        
        return output

class cLN(nn.Module):
    def __init__(self, dimension, eps = 1e-8, trainable=True):
        super(cLN, self).__init__()
        
        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step
        
        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)
        
        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T
        
        entry_cnt = np.arange(channel, channel*(time_step+1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)
        
        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T
        
        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)
        
        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())
