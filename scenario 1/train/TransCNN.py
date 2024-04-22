import os
import pickle
#import rpy2.robjects as robjects
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import scipy.sparse as sp
from torchvision import transforms as tfs
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import HGTConv, Linear
#from torch_geometric.loader import HGTLoader
from torch.cuda.amp import autocast
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
#from torchsummary import summary
import random
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
from torch.utils.data import WeightedRandomSampler
import math
#from einops import rearrange, reduce
#from einops.layers.torch import Rearrange
from torch.nn.parameter import Parameter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        #print(x.size())
        y = self.avg_pool(x)
        #print(y.size())
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        #print(y.size())
        # Multi-scale information fusion
        y = self.sigmoid(y)
        #print(y.size())
        out=x * y.expand_as(x)
        #print(out.size())
        return out


def default(val, d):
    return val if exists(val) else d

def exists(val):
    return val is not None

def conv_kx1(in_channels, out_channels, kernel_size, stride=1):
    layers = []
    padding = kernel_size - 1
    padding_left = padding // 2
    padding_right = padding - padding_left
    layers.append(nn.ConstantPad1d((padding_left, padding_right), 0))
    layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride))
    return nn.Sequential(*layers)

def conv_kx2(in_channels, out_channels, kernel_size, stride=1):
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
    return nn.Sequential(*layers)

class Conv2_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2_Layer, self).__init__()
        #self.relu = nn.ReLU()
        #self.eca1=eca_layer(in_channels,3)
        self.conv1 = conv_kx2(in_channels, out_channels[0], kernel_size)
        self.norm1=nn.BatchNorm2d(out_channels[0])
        self.eca2=eca_layer(out_channels[0],3)
        self.conv2 = conv_kx2(out_channels[0], out_channels[1], kernel_size)
        self.norm2=nn.BatchNorm2d(out_channels[1])
        self.eca3=eca_layer(out_channels[1],3)
        self.relu=nn.GELU()
    def forward(self, x):
        #out = self.eca1(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.eca2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.eca3(out)
        out = self.relu(out)
        return out

def ConvBlock(dim, dim_out = None, kernel_size = 1):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        nn.GELU(),
        conv_kx1(dim, default(dim_out, dim), kernel_size))

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout = 0.3, max_len=35526):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * (-(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False)
        return self.dropout(x)

class gCNN(nn.Module):
    def __init__(self,max_len=2500,dim=128,nhead=4,num_layers=2,pool_size=[(3,1),3,11],out_channels=[7,1],stem_kernel_size=[(7,1),7],dropout_rate=0.3):
        super(gCNN, self).__init__()
        self.stem1 = nn.Sequential(
            Conv2_Layer(15,out_channels,stem_kernel_size[0]),
            nn.MaxPool2d(pool_size[0])
        )
        out_length = np.floor((((max_len-(stem_kernel_size[0][0]*2)+2) - pool_size[0][0]) / pool_size[0][0]) + 1)
        print(out_length)
        self.stem2 = nn.Sequential(
            conv_kx1(4, dim, stem_kernel_size[1]),
            Residual(ConvBlock(dim,dim,stem_kernel_size[1])),
            nn.MaxPool1d(pool_size[1])
        )
        out_length = np.floor(((out_length - pool_size[1]) / pool_size[1]) + 1)
        print(out_length)
        self.stem3 = nn.Sequential(
            #conv_kx1(64, dim, stem_kernel_size[1]),
            Residual(ConvBlock(dim,dim,stem_kernel_size[1])),
            nn.MaxPool1d(kernel_size=pool_size[2],stride=10)
        )
        #out_length = round(((out_length - pool_size[2]) / 1) + 1)
        out_length = np.floor((out_length - pool_size[2]) / 10) + 1
        print(out_length)
        self.position = PositionalEncoding(d_model=dim,max_len=int(out_length))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead,batch_first=True,dim_feedforward=256,activation='gelu',dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.linear = nn.Linear(int(dim * out_length), 1024)
        # self.relu = nn.GELU()
        # self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
    def forward(self, x):
        #print(x.size())
        x = self.stem1(x)
        #print(x.size())
        x=x.view(x.size()[0],x.size()[2],x.size()[3])
        #print(x.size())
        x=x.permute(0,2,1)
        #print(x.size())
        x=self.stem2(x)
        #print(x.size())
        x=self.stem3(x)
        #print(x.size())
        x=x.permute(0, 2, 1)
        #print(x.size())
        x = self.position(x)
        #print(x.size())
        x = self.transformer_encoder(x)
        #print(x.size())
        x = x.reshape(len(x), -1)
        #print(x.size())
        # x = self.linear(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        #print(x.size())
        return x

class mCNN(nn.Module):
    def __init__(self,max_len=28,dim=128,nhead=8,num_layers=2,stem_kernel_size=7,dropout_rate=0.3):
        super(mCNN, self).__init__()
        self.stem = nn.Sequential(
            #nn.Conv1d(4, dim, 7),
            conv_kx1(4, dim, stem_kernel_size),
            Residual(ConvBlock(dim,dim,stem_kernel_size))
            #AttentionPool(dim, pool_size = 2)
        )
        self.position = PositionalEncoding(d_model=dim,max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead,batch_first=True,dim_feedforward=256,activation='gelu',dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.linear = nn.Linear(int(dim * max_len), 1024)
        # self.relu = nn.GELU()
        # self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
    def forward(self, x):
        #print(x.size())
        x=x.permute(0, 2, 1)
        #print(x.size())
        x = self.stem(x)
        #print(x.size())
        #x = self.conv_tower(x)
        #print(x.size())
        x=x.permute(0, 2, 1)
        #print(x.size())
        x = self.position(x)
        #print(x.size())
        x = self.transformer_encoder(x)
        #print(x.size())
        x = x.reshape(len(x), -1)
        #print(x.size())
        # x = self.linear(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        #print(x.size())
        return x

class MLPBilPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout_rate=0.3):
        super(MLPBilPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels[0]))
        for i in range(num_layers - 1):
            self.lins.append(torch.nn.Linear(hidden_channels[i], hidden_channels[i+1]))
        self.bilin = torch.nn.Linear(hidden_channels[-1], hidden_channels[-1], bias=False)
        self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
        # self.bilin = torch.nn.Linear(in_channels,in_channels, bias=False)
        self.relu = nn.GELU()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        self.bilin.reset_parameters()

    def forward(self, x_i, x_j):
        for lin in self.lins:
            x_i, x_j = lin(x_i), lin(x_j)
            x_i, x_j = self.dropout(self.relu(x_i)), self.dropout(self.relu(x_j))
        x = torch.sum(self.bilin(x_i) * x_j, dim=-1)
        # x1 = torch.sum(self.bilin(x_i) * x_j, dim=-1)
        # x2 = torch.sum(self.bilin(x_j) * x_i, dim=-1)
        # x = x1+x2
        # x = torch.sum(x_i * x_j, dim=-1)
        return x,x_i,x_j

# class MLPBilPredictor(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_layers, dropout_rate=0.3):
#         super(MLPBilPredictor, self).__init__()
#         self.lins = torch.nn.ModuleList()
#         self.lins.append(torch.nn.Linear(in_channels, hidden_channels[0]))
#         for i in range(num_layers - 1):
#             self.lins.append(torch.nn.Linear(hidden_channels[i], hidden_channels[i+1]))
#         # self.bilin = torch.nn.Linear(hidden_channels, hidden_channels, bias=False)
#         # self.bilin = torch.nn.Linear(in_channels,in_channels, bias=False)
#         self.pre = torch.nn.Linear(hidden_channels[-1],1)
#         self.relu = nn.GELU()
#         self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
#
#     def reset_parameters(self):
#         for lin in self.lins:
#             lin.reset_parameters()
#         self.lin.reset_parameters()
#
#     def forward(self, x):
#         for lin in self.lins:
#             x= lin(x)
#             x= self.dropout(self.relu(x))
#         x = self.pre(x)
#         # x1 = torch.sum(self.bilin(x_i) * x_j, dim=-1)
#         # x2 = torch.sum(self.bilin(x_j) * x_i, dim=-1)
#         # x = x1+x2
#         # x = torch.sum(x_i * x_j, dim=-1)
#         return x

class CNNmt(torch.nn.Module):
    def __init__(self,max_len=[25,2500],dim=[128,128],nhead=[4,4],num_layers=[2,2],pool_size=[(3,1),3,11],out_channels=[7,1],stem_kernel_size=[(7,1),7],dropout_rate=0.3):
        #max_len=25,dim=128,nhead=8,num_layers=2
        super(CNNmt, self).__init__()
        self.mcnn = mCNN(max_len[0],dim[0],nhead[0],num_layers[0],stem_kernel_size[1],dropout_rate)
        self.gcnn = gCNN(max_len[1],dim[1],nhead[1],num_layers[1],pool_size,out_channels,stem_kernel_size,dropout_rate)
        # self.linear = nn.Linear(3200, 1024)
        # self.relu = nn.GELU()
        # self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
        self.predict=MLPBilPredictor(3584, [1024], 1)
        # self.lin1 = Linear(1024*2, 1024)
        # self.lin2 = Linear(1024,128)
        # self.lin = Linear(128,1)
        # #self.relu = nn.LeakyReLU()
        # self.relu = nn.GELU()
        # #self.sig=nn.Sigmoid()
        # self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
    def encoder(self, xm, xg):
        xm=self.mcnn(xm)
        xg=self.gcnn(xg)
        # xm = self.linear(xm)
        # xm = self.relu(xm)
        # xm = self.dropout(xm)
        # xg = self.linear(xg)
        # xg = self.relu(xg)
        # xg = self.dropout(xg)
        return xm,xg
    def decoder(self, xm, xg):
        #x=torch.cat([xm, xg], dim=1)
        s,xm,xg=self.predict(xm, xg)
        # x=self.lin1(x)
        # x = self.relu(x)dgg
        # x = self.dropout(x)
        # x=self.lin2(x)
        # x = self.relu(x)
        # s=self.lin(x)
        return s,xm,xg
    def forward(self, xm, xg):
        xm,xg=self.encoder(xm,xg)
        s,xm,xg=self.decoder(xm,xg)
        return s,xm,xg

def trainEpoch(model,trainloader,scaler,lossF,device):
    model.train()
    trainloss=0
    auc=0
    aupr=0
    #acc=0
    for step,dat in enumerate(trainloader):
        xm,xg,mid,gid,rel=dat
        xm,xg,mid,gid,rel=xm.to(device),xg.to(device),mid.to(device),gid.to(device),rel.to(device)
        optimizer.zero_grad()
        # 训练模型
        #with autocast():
        #out = model(dat.x_dict,dat.seq_dict, dat.edge_index_dict)
        out,_,_ = model(xm, xg)
        out=out.view(-1)
        #pre=out[mid,gid]
        #print(rel)
        #loss = F.binary_cross_entropy_with_logits(out,rel)
        loss=lossF(out,rel)
        #metric=model_evaluation(pre,rel.int())
        #print(out)
        pre=torch.sigmoid(out)
        #print(pre)
        #assert torch.isnan(loss).sum() == 0, print(loss)
        metric0=roc_auc_score(rel.cpu().detach().numpy(), pre.cpu().detach().numpy())
        metric1= average_precision_score(rel.cpu().detach().numpy(), pre.cpu().detach().numpy())
        #         scaler.scale(loss).backward()
        #         scaler.step(optimizer)  # optimizer.step
        #         scaler.update()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trainloss = trainloss+loss.item()
        #acc=acc+metric['accuracy']
        auc=auc+metric0
        aupr=aupr+metric1
        # if (step % 100 == 0):
        #     train_loss = (trainloss / (step+1))
        #     auc_batch = (auc / (step+1))
        #     aupr_batch = (aupr / (step+1))
        #     #train_loss.append(trainloss)
        #     print('Batch:',step,train_loss,auc_batch,aupr_batch)
    return (trainloss/(step+1),auc/(step+1),aupr/(step+1))

def evaluate(model,valloader,lossF,device):
    model.eval()
    valloss=0
    valauc=0
    valaupr=0
    with torch.no_grad():
        for step,dat in enumerate(valloader):
            xm,xg,mid,gid,rel=dat
            xm,xg,mid,gid,rel=xm.to(device),xg.to(device),mid.to(device),gid.to(device),rel.to(device)
            #pre=out[mid,gid]
            out,_,_ = model(xm, xg)
            out=out.view(-1)
            #loss = F.binary_cross_entropy_with_logits(out,rel)
            loss=lossF(out,rel)
            #metric=model_evaluation(pre,rel.int())
            #print(rel,pre)
            pre=torch.sigmoid(out)
            auc=roc_auc_score(rel.cpu(), pre.cpu())
            aupr= average_precision_score(rel.cpu(), pre.cpu())
            valloss = valloss+loss.item()
            valauc=valauc+auc
            valaupr=valaupr+aupr
            # if (step % 100 == 0):
            #     val_loss = (valloss / (step+1))
            #     #val_loss.append(valloss)
            #     val_auc=(valauc/(step+1))
            #     val_aupr=(valaupr/(step+1))
            #     print('Batch:',step,val_loss,val_auc,val_aupr)
    return (valloss/(step+1)),(valauc/(step+1)),(valaupr/(step+1))

class seqData(Dataset):
    def __init__(self,xm,xg,label,edgeidx):
        self.source= xm
        self.target=xg
        self.edge=edgeidx
        self.label = label
        self.length = len(self.label)
    def __getitem__(self, index):
        s=self.edge[0][index]
        t=self.edge[1][index]
        return self.source[s], self.target[t],s,t,self.label[index]
    def __len__(self):
        return self.length

class FocalLoss(nn.Module):
    def __init__(self,alpha=0.25,gamma=2.0,reduce='mean'):
        super(FocalLoss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self,classifications,targets):
        # classifcation:[N,K]
        # targets: [N,K]的one-hot编码
        alpha = self.alpha
        gamma = self.gamma
        # classifications = classifications.view(-1)  # 不经过sigmoid的classification；
        # targets = targets.view(-1)                  # 应该是 one-hot
        # ce_loss: 对应公式中 -log(pt),也就是普通的 交叉熵损失；--> 该函数接收未经sigmoid的函数；
        ce_loss = F.binary_cross_entropy_with_logits(classifications, targets, reduction="none")
        #focal loss
        p = torch.sigmoid(classifications)                # 经过sigmoid
        p_t = p * targets + (1 - p) * (1 - targets)       #  计算pt
        loss = ce_loss * ((1 - p_t) ** gamma)             # -log(pt) * (1-pt) ** ganmma
        if alpha >= 0:
            # 对应公式中alpha_t控制损失的权重
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets) # 和pt求解过程一样
            loss = alpha_t * loss                         # 最终focal loss
        if self.reduce=='sum':
            loss = loss.sum()
        elif self.reduce=='mean':
            loss = loss.mean()
        else:
            raise ValueError('reduce type is wrong!')
        return loss

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1-self.alpha)*pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        #         loss = - self.alpha * pt ** self.gamma * target * torch.log(pt) - \
        #             (1-self.alpha)*(1 - pt)** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

# class EarlyStopping():
#     def __init__(self, tolerance=5, min_delta=0.1):
#         self.tolerance = tolerance
#         self.min_delta = min_delta
#         self.counter = 0
#         self.early_stop = False
#     def __call__(self, train_auc, validation_auc):
#         if (train_auc - validation_auc) > self.min_delta:
#             self.counter +=1
#             if self.counter >= self.tolerance:
#                 self.early_stop = True

class EarlyStopping():
    def __init__(self, tolerance=50, min_delta=0.1):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
    def __call__(self, val_best_auc, val_auc):
        if val_auc<val_best_auc:
            self.counter +=1
            if self.counter >= self.tolerance:
                self.early_stop = True

if __name__ == '__main__':

    torch.cuda.set_device(2)
    set_seed(2022)
    with open('dataSplit_negall.pkl','rb') as f:  # Python 3: open(..., 'rb')
        Xm,Xg,nodeM,nodeG,trainidx,validx,testidx,trainId,trainLabel,testId,testLabel,valId,valLabel,ppi,m,g = pickle.load(f)

    train_data=seqData(Xm,Xg,trainLabel,trainId)
    #test_data=seqData(Xm,Xg,testLabel,testId)
    val_data=seqData(Xm,Xg,valLabel,valId)

    train_loader = DataLoader(dataset=train_data, batch_size=1024*2, shuffle=True,num_workers=8,pin_memory=True)
    val_loader = DataLoader(dataset=val_data, batch_size=1024*2, shuffle=True,num_workers=8,pin_memory=True)
    #test_loader = DataLoader(dataset=test_data, batch_size=1024*2, shuffle=True,num_workers=8, pin_memory=True)

    model = CNNmt(max_len=[28,2500],dim=[128,128],nhead=[8,8],num_layers=[1,1],pool_size=[(3,1),3,1],out_channels=[7,1],stem_kernel_size=[(7,1),3],dropout_rate=0.3)
    print(model)

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device='cuda:2'
    model.to(device)

    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            torch.nn.init.kaiming_normal_(m.weight, mode = 'fan_in')

    # for m in model.modules():
    #     if isinstance(m, (torch.nn.Linear)):
    #         torch.nn.init.kaiming_normal_(m.weight, mode = 'fan_in')

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=5e-3)#
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=0.000001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',factor=0.5,verbose=1,min_lr=0.000001,patience=20)
    lossF=FocalLoss(alpha=0.65, gamma=2,reduce='mean')
    #lossF=FocalLoss(alpha=-1, gamma=0.2,reduce='mean')
    #lossF=BCEFocalLoss(gamma=1, alpha=0.75, reduction='elementwise_mean')
    early_stopping = EarlyStopping(tolerance=20, min_delta=0.15)

    # Define the early stopping parameters
    patience = 50
    #best_va = float('inf')
    counter = 0
    
    import time
    best_val_auc= best_val_aupr= 0
    best_epoch=-1
    trainloss=[]
    valloss=[]
    valauc=[]
    trainauc=[]
    valaupr=[]
    trainaupr=[]
    lrchange=[]
    for epoch in range(1, 1001):
        since = time.time()
        print('{} optim: {}'.format(epoch, optimizer.param_groups[0]['lr']))
        lrchange.append(optimizer.param_groups[0]['lr'])
        train_loss,train_auc,train_aupr = trainEpoch(model,train_loader,scaler,lossF,device)
        trainloss.append(train_loss)
        print('train_loss:',train_loss)
        val_loss,val_auc,val_aupr = evaluate(model,val_loader,lossF,device)
        valauc.append(val_auc)
        valloss.append(val_loss)
        trainauc.append(train_auc)
        valaupr.append(val_aupr)
        trainaupr.append(train_aupr)
        #scheduler.step() 
        scheduler.step(val_aupr)
        if val_aupr > best_val_aupr:
            best_val_auc = val_auc
            best_val_aupr=val_aupr
            counter = 0
            best_epoch = epoch
            #print(f'best aupr {epoch}')
            torch.save(model, './TransCNN/seqCNNbest_2.pt')
            time_elapsed = time.time() - since
            log = 'Epoch: {:03d}, Epoch complete in {:.0f}m {:.0f}s, trainLoss: {:.4f}, Valloss: {:.4f}, Trainauc: {:.4f}, Valauc: {:.4f}, Valbestauc: {:.4f},Trainaupr: {:.4f}, Valaupr: {:.4f}, Valbestaupr: {:.4f}'
            print(log.format(epoch, time_elapsed // 60, time_elapsed % 60,train_loss, val_loss,train_auc,val_auc,best_val_auc,train_aupr,val_aupr,best_val_aupr))
        else:
            counter += 1
            time_elapsed = time.time() - since
            log = 'Epoch: {:03d}, Epoch complete in {:.0f}m {:.0f}s, trainLoss: {:.4f}, Valloss: {:.4f}, Trainauc: {:.4f}, Valauc: {:.4f}, Valbestauc: {:.4f},Trainaupr: {:.4f}, Valaupr: {:.4f}, Valbestaupr: {:.4f}'
            print(log.format(epoch, time_elapsed // 60, time_elapsed % 60,train_loss, val_loss,train_auc,val_auc,best_val_auc,train_aupr,val_aupr,best_val_aupr))
            if counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                print(f'best aupr at epoch {best_epoch}')
                break
        
        # if val_aupr > best_val_aupr:
        #     best_val_auc = val_auc
        #     best_val_aupr=val_aupr
        #     torch.save(model, '/mnt/sda/liupei/miRGraph/TransCNN/seqCNNbest.pt')
        # #scheduler.step()
        # time_elapsed = time.time() - since
        # log = 'Epoch: {:03d}, Epoch complete in {:.0f}m {:.0f}s, trainLoss: {:.4f}, Valloss: {:.4f}, Trainauc: {:.4f}, Valauc: {:.4f}, Valbestauc: {:.4f},Trainaupr: {:.4f}, Valaupr: {:.4f}, Valbestaupr: {:.4f}'
        # print(log.format(epoch, time_elapsed // 60, time_elapsed % 60,train_loss, val_loss,train_auc,val_auc,best_val_auc,train_aupr,val_aupr,best_val_aupr))


    with open('./TransCNN/seqCNNResult_2.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([trainloss, trainauc,trainaupr,valloss,valauc,valaupr,lrchange], f)

    torch.save(model, './TransCNN/seqCNN_2.pt')