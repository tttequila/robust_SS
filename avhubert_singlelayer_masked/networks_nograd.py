import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import model_builder
from torch.autograd import Variable

# Model framework
class AudioVisualModel(nn.Module):
    def __init__(self, enc_dim=512, feature_dim=256, hidden_dim=128, sr=16000, win=2, layer=8, R_a=1, R_av=3, 
                 kernel=3, causal=False, requires_grad_pretrained=True,
                 weights_lip_reading='/home/liuqinghua/code/code_space/realtime_extract/MuSE_demo-master/model/visual_frontend.pt',
                 avHuBERT_path='/mntnfs/lee_data1/chenshutang/mask_avhubert/av_hubert/avhubert',
                 weights_avHuBERT='/mntnfs/lee_data1/chenshutang/pretrained/avhubert/base_vox_iter5_clean.pt'):
        super(AudioVisualModel, self).__init__()
        
        # hyper parameters
        self.num_spks = 2
        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim    
        self.win = int(sr*win/1000)
        self.stride = self.win // 2       
        self.layer = layer
        self.R_a = R_a
        self.R_av = R_av
        self.kernel = kernel
        self.causal = causal
        self.requires_grad_pretrained = requires_grad_pretrained
        self.weights_lip_reading = weights_lip_reading # checkpoint path
        self.avHuBERT_path = avHuBERT_path
        self.weights_avHuBERT = weights_avHuBERT

        # # loading pretrained model
        # self.visual_front_end = model_builder.lip_reading_net(weights=self.weights_lip_reading)
        # for p in self.parameters():
        #     p.requires_grad = self.requires_grad_pretrained

        # audio encoder
        self.audio_enc = nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.stride)

        # visual encoder
        self.avHuBERT = model_builder.av_hubert_net(user_dir=self.avHuBERT_path, weights=self.weights_avHuBERT)
        self.avhubert_adpt = model_builder.AVHuBERT_Adapter(B=768, H=256, Num_Layers=1)
        # self.visual_adapt = model_builder.videoEncoder(self.feature_dim, self.feature_dim*2, 5)

        # separator
        self.separator = model_builder.Separator_TCN(self.enc_dim, self.enc_dim, self.feature_dim, self.feature_dim*4,
                                    8, self.R_a, self.R_av, self.kernel, causal=self.causal)

        # decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)
        
    def forward(self, mixture, visual):
        # padding
        output, rest = self.pad_signal(mixture)
        
        # waveform encoder
        enc_output = self.audio_enc(output)  # B, N, L
        batch_size, _, L = enc_output.shape
        
        # # frame encoder
        # visual_emb = self.visual_front_end(visual)   # B, num_frame, 256 
        # visual_emb = self.visual_adapt(visual_emb)
        # visual_emb = F.interpolate(visual_emb, L, mode='linear', align_corners=False)
        
        visual = visual.unsqueeze(dim=1)
        
        # new frame encoder
        with torch.no_grad():
            visual_emb = model_builder.av_hubert_feature(self.avHuBERT, visual, audio=None, output_layer=None)
        visual_emb = self.avhubert_adpt(visual_emb)
        
        # lip movement resampling
        visual_emb = F.interpolate(visual_emb, L, mode='linear', align_corners=False)

        # generate masks
        mask = self.separator(enc_output, visual_emb).view(batch_size, self.enc_dim, -1)  # B, N, L
        masked_output = enc_output*mask
        
        # waveform decoder
        output = self.decoder(masked_output.view(batch_size, self.enc_dim, -1))  # B, 1, L
        output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B, 1, L
        output = output.view(batch_size, -1)  # B, T
        
        return output

    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        
        pad_aux = Variable(torch.zeros(batch_size, 1, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

class AudioOnlyModel(nn.Module):
    def __init__(self, enc_dim=512, feature_dim=128, hidden_dim=128, sr=16000, win=2, layer=8, R=6,  
                 kernel=3, causal=False):
        super(AudioOnlyModel, self).__init__()
        
        # hyper parameters
        self.num_spks = 2
        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim    
        self.win = int(sr*win/1000)
        self.stride = self.win // 2       
        self.layer = layer
        self.R = R
        self.kernel = kernel
        self.causal = causal
        
        # encoder
        self.audio_enc = nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.stride)

        # separator
        self.separator = model_builder.Separator_blind_source(self.enc_dim, self.feature_dim, self.hidden_dim, 100, self.R)
        
        # mask_generator
        self.mask_generator = nn.Conv1d(self.enc_dim, self.enc_dim*2, 1)

        # decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)
        
    def forward(self, input):
        mixture = input['audio_mix']

        # padding
        output, rest = self.pad_signal(mixture)
        
        # waveform encoder
        enc_output = self.audio_enc(output)  # B, N, L
        batch_size, _, L = enc_output.shape

        # generate masks
        masks = self.separator(enc_output).view(batch_size, self.num_spks, self.enc_dim, -1)  # B, num_spk, N, L
        masked_output = enc_output.unsqueeze(1) * masks
        
        # waveform decoder
        output = self.decoder(masked_output.view(batch_size*self.num_spks, self.enc_dim, -1))  # B*num_spk, 1, L
        output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*num_spk, 1, L
        output = output.view(batch_size,self.num_spks, -1)  # B, num_spk, T
        
        return output

    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        
        pad_aux = Variable(torch.zeros(batch_size, 1, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest



def check_parameters(net):
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6

def test_model():
    x = torch.rand(2, 48000)
    visual = torch.rand(2, 75, 112, 112)
    nnet = AudioVisualModel()
    x = nnet(x, visual)
    print("param:", str(check_parameters(nnet))+' Mb')
    for name, param in nnet.named_parameters():
        if not param.requires_grad:
            print(name)

if __name__ == "__main__":
    test_model()
