#### ReduceLROnPlateau####
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
from torch_geometric.loader import HGTLoader
from torch_geometric.loader import LinkNeighborLoader
from torch.cuda.amp import autocast
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
#from torchsummary import summary
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers,node_types,metadata):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels[0])
        # self.lin_dict['miRNA']=Linear(2656, hidden_channels[0])
        # self.lin_dict['gene']=Linear(18454, hidden_channels[0])
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HGTConv(hidden_channels[i], hidden_channels[i+1], metadata,num_heads)
            self.convs.append(conv)
        # self.lin1 = Linear(hidden_channels[-1], out_channels)
        # self.lin2 = Linear(hidden_channels[-1], out_channels)
        self.relu = nn.GELU()
    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        # xm=self.relu(self.lin1(x_dict['miRNA']))
        # xg=self.relu(self.lin2(x_dict['gene']))
        xm=x_dict['miRNA']
        xg=x_dict['gene']
        return (xm,xg)

# class MLPBilPredictor(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_layers, dropout_rate=0.3):
#         super(MLPBilPredictor, self).__init__()
#         self.lins = torch.nn.ModuleList()
#         self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
#         for _ in range(num_layers - 1):
#             self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
#         self.bilin = torch.nn.Linear(hidden_channels, hidden_channels, bias=False)
#         self.relu = nn.GELU()
#
#     def reset_parameters(self):
#         for lin in self.lins:
#             lin.reset_parameters()
#         self.bilin.reset_parameters()
#
#     def forward(self, x_i, x_j):
#         for lin in self.lins:
#             x_i, x_j = lin(x_i), lin(x_j)
#             x_i, x_j = self.relu(x_i), self.relu(x_j)
#         x = torch.sum(self.bilin(x_i) * x_j, dim=-1)
#         # x2 = torch.sum(self.bilin(x_j) * x_i, dim=-1)
#         # x = x1+x2
#         #x = torch.sum(x_i * x_j, dim=-1)
#         return x

class MLPBilPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout_rate=0.3):
        super(MLPBilPredictor, self).__init__()
        # self.lins = torch.nn.ModuleList()
        # self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        # for _ in range(num_layers - 1):
        #     self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.bilin = torch.nn.Linear(hidden_channels, hidden_channels, bias=False)
        # self.bilin = torch.nn.Linear(in_channels,in_channels, bias=False)
        # self.relu = nn.GELU()

    def reset_parameters(self):
        # for lin in self.lins:
        #     lin.reset_parameters()
        self.bilin.reset_parameters()

    def forward(self, x_i, x_j):
        # for lin in self.lins:
        #     x_i, x_j = lin(x_i), lin(x_j)
        #     x_i, x_j = self.relu(x_i), self.relu(x_j)
        x = torch.sum(self.bilin(x_i) * x_j, dim=-1)
        # x1 = torch.sum(self.bilin(x_i) * x_j, dim=-1)
        # x2 = torch.sum(self.bilin(x_j) * x_i, dim=-1)
        # x = x1+x2
        # x = torch.sum(x_i * x_j, dim=-1)
        return x

class HGTmt(torch.nn.Module):
    def __init__(self,hidden_channels, out_channels, num_heads, num_layers,node_types,metadata):
        super(HGTmt, self).__init__()
        self.hgt=HGT(hidden_channels, out_channels, num_heads, num_layers,node_types,metadata)
        self.predict=MLPBilPredictor(hidden_channels[-1], out_channels, 1, 0.3)

    def encoder(self,x_dict,edge_index_dict):
        xm,xg=self.hgt(x_dict,edge_index_dict)
        #print(xm.size(),xg.size())
        return xm,xg

    def decoder(self,xm,xg,edge):
        xm=xm[edge[0]]
        xg=xg[edge[1]]
        s=self.predict(xm,xg)
        return s

    def forward(self,x_dict,edge_index_dict,label_edge):
        xm,xg=self.encoder(x_dict,edge_index_dict)
        s=self.decoder(xm,xg,label_edge)
        return s

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

def trainEpoch(model,trainloader,scaler,lossF,device,xm,xg):
    model.train()
    trainloss=0
    auc=0
    aupr=0
    #acc=0
    for step,dat in enumerate(trainloader):
        #print(device)
        #dat=dat.to(device)
        # dat['miRNA'].x=xm
        # dat['gene'].x=xg
        dat=T.ToUndirected()(dat)
        dat=dat.to(device)
        #print(dat)
        out = model(dat.x_dict,dat.edge_index_dict,dat['regulate'].edge_label_index)
        out=out.view(-1)
        #print(out.size())
        rel=dat['regulate'].edge_label
        #print(rel.size())
        #loss = F.binary_cross_entropy_with_logits(out,rel)
        loss=lossF(out,rel)
        #metric=model_evaluation(pre,rel.int())
        #print(out)
        pre=torch.sigmoid(out)
        #print(pre.size())
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

def evaluate(model,valloader,lossF,device,xm,xg):
    model.eval()
    valloss=0
    valauc=0
    valaupr=0
    with torch.no_grad():
        for step,dat in enumerate(valloader):
            #print(device)
            # dat['miRNA'].x=xm
            # dat['gene'].x=xg
            dat=T.ToUndirected()(dat)
            #print(step)
            dat=dat.to(device)
            out = model(dat.x_dict,dat.edge_index_dict,dat['regulate'].edge_label_index)
            out=out.view(-1)
            #print(out.size())
            rel=dat['regulate'].edge_label
            #print(rel.size())
            #loss = F.binary_cross_entropy_with_logits(out,rel)
            loss=lossF(out,rel)
            #metric=model_evaluation(pre,rel.int())
            #print(rel,pre)
            pre=torch.sigmoid(out)
            #print(pre.size())
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

    torch.cuda.set_device(0)
    set_seed(2022)
        
    with open('dataCombine_negall.pkl','rb') as f:  # Python 3: open(..., 'rb')
        train_data,val_data,_,m,g = pickle.load(f)

    train_data ['miRNA'].sim = train_data ['miRNA'].mm + train_data ['miRNA'].x
    val_data ['miRNA'].sim = train_data ['miRNA'].mm + train_data ['miRNA'].x
    #test_data ['miRNA'].sim = train_data ['miRNA'].mm + train_data ['miRNA'].x
    train_data ['gene'].sim = train_data ['gene'].gg + train_data ['gene'].x
    val_data ['gene'].sim = train_data ['gene'].gg + train_data ['gene'].x
    #test_data ['gene'].sim = train_data ['gene'].gg + train_data ['gene'].x
    nodetypes=train_data.node_types
    metadata=train_data.metadata()

    train_data['miRNA'].x=train_data['miRNA'].sim
    train_data['gene'].x=train_data['gene'].sim
    val_data['miRNA'].x=val_data['miRNA'].sim
    val_data['gene'].x=val_data['gene'].sim

    train_data['miRNA'].num_nodes = len(m)
    train_data['gene'].num_nodes = len(g)
    val_data['miRNA'].num_nodes = len(m)
    val_data['gene'].num_nodes = len(g)

    del train_data['miRNA'].sim
    del train_data['miRNA'].mm
    del train_data['miRNA'].seq
    del train_data['gene'].sim
    del train_data['gene'].gg
    del train_data['gene'].seq
    del train_data['rev_regulate']

    del val_data['miRNA'].sim
    del val_data['miRNA'].mm
    del val_data['miRNA'].seq
    del val_data['gene'].sim
    del val_data['gene'].gg
    del val_data['gene'].seq
    del val_data['rev_regulate']

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors={('miRNA','regulate','gene'):[8]*4,('miRNA','cofamily','miRNA'):[8]*4,('gene','coocurrence','gene'):[8]*4},
        #num_neighbors={('miRNA','regulate','gene'):[-1]*1,('miRNA','cofamily','miRNA'):[-1]*1,('gene','coocurrence','gene'):[-1]*1},
        #edge_label_index=('miRNA','regulate','gene'),
        #num_neighbors=[-1]*1,
        #num_neighbors=[8]*4,
        edge_label_index=(('miRNA','regulate','gene'),train_data['regulate'].edge_label_index),
        edge_label=train_data['regulate'].edge_label,
        batch_size=1024*2,
        #weight_attr=None,
        shuffle=True,
        #num_workers=8, 
        pin_memory=True,
    )
    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors={('miRNA','regulate','gene'):[8]*4,('miRNA','cofamily','miRNA'):[8]*4,('gene','coocurrence','gene'):[8]*4},
        #num_neighbors={('miRNA','regulate','gene'):[-1]*1,('miRNA','cofamily','miRNA'):[-1]*1,('gene','coocurrence','gene'):[-1]*1},
        #edge_label_index=('miRNA','regulate','gene'),
        #num_neighbors=[-1]*1,
        #num_neighbors=[8]*4,
        edge_label_index=(('miRNA','regulate','gene'),val_data['regulate'].edge_label_index),
        edge_label=val_data['regulate'].edge_label,
        batch_size=1024*2,
        #weight_attr=None,
        shuffle=True,
        #num_workers=8, 
        pin_memory=True,
    )

    xm=train_data['miRNA'].x
    xg=train_data['gene'].x
    model = HGTmt(hidden_channels=[1024,256,128], out_channels=128, num_heads=8, num_layers=2,node_types=nodetypes,metadata=metadata)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device='cpu'
    device='cuda:0'
    #print(device)
    model.to(device)
    print(model)
    
    #torch.cuda.set_device(1)
    #set_seed(2022)
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear)):
            torch.nn.init.kaiming_normal_(m.weight, mode = 'fan_in')
    #weight_decay=5e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    #0.1 0.05 0.01 0.005 0.0001
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
    #print(device)
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
        train_loss,train_auc,train_aupr = trainEpoch(model,train_loader,scaler,lossF,device,xm,xg)
        trainloss.append(train_loss)
        print('train_loss:',train_loss)
        val_loss,val_auc,val_aupr = evaluate(model,val_loader,lossF,device,xm,xg)
        valauc.append(val_auc)
        valloss.append(val_loss)
        trainauc.append(train_auc)
        valaupr.append(val_aupr)
        trainaupr.append(train_aupr)
        #time_elapsed = time.time() - since1
        #print('Val and Testing in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        #scheduler.step() 
        scheduler.step(val_aupr)
        if val_aupr > best_val_aupr:
            best_val_auc = val_auc
            best_val_aupr=val_aupr
            counter = 0
            best_epoch = epoch
            #print(f'best aupr {epoch}')
            torch.save(model, './HGT/HGTbest_linkloader_ReduceLR_2.pt')
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
                
        # scheduler.step() 
        # #scheduler.step(val_aupr)
        # time_elapsed = time.time() - since
        # log = 'Epoch: {:03d}, Epoch complete in {:.0f}m {:.0f}s, trainLoss: {:.4f}, Valloss: {:.4f}, Trainauc: {:.4f}, Valauc: {:.4f}, Valbestauc: {:.4f},Trainaupr: {:.4f}, Valaupr: {:.4f}, Valbestaupr: {:.4f}'
        # print(log.format(epoch+1, time_elapsed // 60, time_elapsed % 60,train_loss, val_loss,train_auc,val_auc,best_val_auc,train_aupr,val_aupr,best_val_aupr))

    with open('./HGT/HGTResult_linkloader_ReduceLR_2.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([trainloss, trainauc,trainaupr,valloss,valauc,valaupr,lrchange], f)

    torch.save(model, './HGT/HGT_linkloader_ReduceLR_2.pt')