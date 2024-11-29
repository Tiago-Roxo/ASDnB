import torch
import torch.nn as nn

from model.Classifier import BGRU
from model.Encoder import visual_encoder
from model.audioEncoder import audioEncoder

N = 128

class ASD_Model(nn.Module):
    def __init__(self):
        super(ASD_Model, self).__init__()
        
        self.visualEncoder  = visual_encoder()
        self.audioEncoder  = audioEncoder(layers = [3, 4, 6, 3],  num_filters = [16, 32, 64, 128])
        self.GRU = BGRU(N)

   
    def forward_visual_frontend(self, xf, xb):
        B, T, W, H = xf.shape  
        xf = xf.view(B, 1, T, W, H)
        xf = (xf / 255 - 0.4161) / 0.1688
        xb = xb.view(B, 1, T, W, H)
        xb = (xb / 255 - 0.4161) / 0.1688
        x = self.visualEncoder(xf, xb)
        return x

    def forward_audio_frontend(self, x):    
        x = x.unsqueeze(1).transpose(2, 3)     
        x = self.audioEncoder(x)
        return x

    def forward_combination_backend(self, x1, x2):  
        x = x1 + x2
        x = self.GRU(x)   
        x = torch.reshape(x, (-1, N))
        return x 
    

    def forward_data_backend(self, x):  
        x = torch.reshape(x, (-1, N))
        return x