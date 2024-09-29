import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, AutoModel

class MERT_Classifier(nn.Module):
    def __init__(self, n_class, freeze=False):
        super().__init__()
        #Load the pretrained model and weight of MERT
        self.mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)

        #Freeze (Bashed by user)
        if freeze:
            for param in self.mert_model.parameters():
                param.requires_grad = False

        #Classifier (composed by two linear layer)
        self.classifier = nn.Sequential(nn.Linear(1024,256),     #Output shape of MERT are 1024
                                        nn.BatchNorm1d(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(256,n_class))    
    def forward(self,x):
        #Process input data
        inputs = self.processor(x, sampling_rate=24000, return_tensors="pt", padding=True)
        inputs = inputs["input_values"].squeeze(0).to('cuda')
        
        #Feature extractor output and pooling along time axis
        mert_output = self.mert_model(inputs,output_hidden_states=False)
        mert_output = torch.mean(mert_output.last_hidden_state, dim=1)      #Pooling
        
        #Output of the two linear layer (Logits) 
        out = self.classifier(mert_output)
        return out