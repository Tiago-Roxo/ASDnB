import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, subprocess, pandas, tqdm
import pandas as pd
from subprocess import PIPE

from loss import lossAV, lossV, lossA, lossVB
from model.Model import ASD_Model

import warnings
warnings.filterwarnings("ignore")
device="cuda"

class ASDnB(nn.Module):
    def __init__(self, lr = 0.001, lrDecay = 0.95, **kwargs):
        super(ASDnB, self).__init__()        
        self.model = ASD_Model().cuda(device)
        self.lossAV = lossAV().cuda(device)
        self.lossV = lossV().cuda(device)
        self.lossA = lossA().cuda(device)
        self.lossVB = lossVB().cuda(device)
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1000 / 1000))

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)  # StepLR
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']

        alpha = 0.5 + 1/60 * (epoch - 1)
        beta = 0.5 - 1/60 * (epoch - 1)

        for num, (audioFeature, visualFeature, visualFeatureBody, labels) in enumerate(loader, start=1):
            self.zero_grad()

            audioEmbed = self.model.forward_audio_frontend(audioFeature[0].cuda(device))
            visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda(device), visualFeatureBody[0].cuda(device)) 

            outsAV  = self.model.forward_combination_backend(audioEmbed, visualEmbed)  
            outsV = self.model.forward_data_backend(visualEmbed)

            labels = labels[0].reshape((-1)).cuda(device) # Loss

            nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels)
            nlossV = self.lossV.forward(outsV, labels)
            nloss = alpha * nlossAV + beta * nlossV

            loss += nloss.detach().cpu().numpy()
            top1 += prec
            nloss.backward()
            self.optim.step()
            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), 100 * (top1/index)))
            sys.stderr.flush()  

        sys.stdout.write("\n")      

        return loss/num, lr

    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        predScores = []
        for audioFeature, visualFeature, visualFeatureBody, labels in tqdm.tqdm(loader):
            with torch.no_grad():                
                audioEmbed  = self.model.forward_audio_frontend(audioFeature[0].cuda(device))
                visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda(device), visualFeatureBody[0].cuda(device))

                outsAV = self.model.forward_combination_backend(audioEmbed, visualEmbed) 

                labels = labels[0].reshape((-1)).cuda(device)             
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels)    
                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)

        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = []
        labels = pd.Series( ['SPEAKING_AUDIBLE' for line in evalLines])
        scores = pd.Series(predScores)
        evalRes = pd.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1,inplace=True)
        evalRes.drop(['instance_id'], axis=1,inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python3 -O utils/get_ava_active_speaker_performance.py -g %s -p %s "%(evalOrig, evalCsvSave)
        mAP = float(str(subprocess.run(cmd, shell=True, stdout=PIPE, stderr=PIPE).stdout).split(' ')[2][:5])
        return mAP

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path, map_location=device)
        for name, param in loadedState.items():
            origName = name;
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)