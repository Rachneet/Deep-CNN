import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pretrained_vectors import *


class DPCNN(nn.Module):

    def __init__(self,weights_matrix):

        super(DPCNN,self).__init__()
        self.channel_size =250
        #self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix)
        emb_wts = torch.Tensor(weights_matrix)
        self.embedding =nn.Embedding(*emb_wts.shape)
        self.embedding.weight= nn.Parameter(emb_wts)
        # regard the word vector as one channel graph
        self.conv_region_embedding = nn.Conv2d(1, self.channel_size,(3, 300), stride =1)
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3,1), stride =1)
        self.pooling = nn.MaxPool2d(kernel_size=(3,1), stride=2) #downsampling /2
        self.padding_conv = nn.ZeroPad2d((0,0,1,1))
        self.padding_pool = nn.ZeroPad2d((0,0,0,1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(self.channel_size, 2)
        self.sigmoid = nn.Sigmoid()




    def forward(self, x):
        #batch = x.shape[0]
        #print(batch)
        embedded = self.embedding(x)
        batch, width, height = embedded.shape
        embedded = embedded.view((batch,1,width,height))
        # Region embedding
        x = self.conv_region_embedding(embedded)        # [batch_size, channel_size, length, 1]

        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)

        while x.size()[-2] > 2:
            x = self._block(x)

        x = x.view(batch, self.channel_size)
        x = self.linear_out(x)
        x = self.sigmoid(x)

        return x

    def _block(self, x):
        # Pooling
        x = self.padding_pool(x)
        px = self.pooling(x)

        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)

        # Short Cut
        x = x + px

        return x



