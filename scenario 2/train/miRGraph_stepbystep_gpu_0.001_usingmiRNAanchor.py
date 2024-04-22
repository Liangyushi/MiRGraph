import os
import pickle
# import rpy2.robjects as robjects
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
# from torch_geometric.loader import HGTLoader
from torch.cuda.amp import autocast
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
#from torchsummary import summary
import random
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
from torch.utils.data import WeightedRandomSampler
import math
from torch.nn.parameter import Parameter


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLPBilPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout_rate=0.3):
        super(MLPBilPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels[0]))
        for i in range(num_layers - 1):
            self.lins.append(torch.nn.Linear(hidden_channels[i], hidden_channels[i + 1]))
        self.bilin = torch.nn.Linear(hidden_channels[-1], hidden_channels[-1], bias=False)
        self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
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
        return x, x_i, x_j


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

class HerGraph(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(HerGraph, self).__init__()
        self.pre = MLPBilPredictor(3584 + 128, [1024], 1, 0.3)
        self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
        self.relu = nn.GELU()

    def decoderMLP(self, xm, xg):
#         xm = xm[edge[0]]
#         xg = xg[edge[1]]
        s, xm, xg = self.pre(xm, xg)
        return s, xm, xg

    def forward(self, xm1,xg1,xm2,xg2):
#         xm1 = seq_dict['miRNA']
#         xg1 = seq_dict['gene']
#         xm2 = x_dict['miRNA']
#         xg2 = x_dict['gene']
#         xm = self.dropout(self.relu(torch.cat([xm1, xm2], dim=1)))
#         xg = self.dropout(self.relu(torch.cat([xg1, xg2], dim=1)))
        xm = self.relu(torch.cat([xm1, xm2], dim=1))
        xg = self.relu(torch.cat([xg1, xg2], dim=1))
        s, xm, xg = self.decoderMLP(xm, xg)
        return s, xm, xg


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduce='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, classifications, targets):
        # classifcation:[N,K]
        # targets: [N,K]的one-hot编码
        alpha = self.alpha
        gamma = self.gamma
        # classifications = classifications.view(-1)  # 不经过sigmoid的classification；
        # targets = targets.view(-1)                  # 应该是 one-hot
        # ce_loss: 对应公式中 -log(pt),也就是普通的 交叉熵损失；--> 该函数接收未经sigmoid的函数；
        ce_loss = F.binary_cross_entropy_with_logits(classifications, targets, reduction="none")
        # focal loss
        p = torch.sigmoid(classifications)  # 经过sigmoid
        p_t = p * targets + (1 - p) * (1 - targets)  # 计算pt
        loss = ce_loss * ((1 - p_t) ** gamma)  # -log(pt) * (1-pt) ** ganmma
        if alpha >= 0:
            # 对应公式中alpha_t控制损失的权重
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)  # 和pt求解过程一样
            loss = alpha_t * loss  # 最终focal loss
        if self.reduce == 'sum':
            loss = loss.sum()
        elif self.reduce == 'mean':
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
               (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        #         loss = - self.alpha * pt ** self.gamma * target * torch.log(pt) - \
        #             (1-self.alpha)*(1 - pt)** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


# def trainAll(model, train_data, lossF):
#     model.train()
#     out, _, _ = model(train_data.seq_dict, train_data.sim_dict, train_data.edge_index_dict,
#                       train_data['regulate'].edge_label_index)
#     out = out.view(-1)
#     # print('out')
#     rel = train_data['regulate'].edge_label
#     # loss = F.binary_cross_entropy_with_logits(out,rel)
#     loss = lossF(out, rel)
#     pre = torch.sigmoid(out)
#     assert torch.isnan(loss).sum() == 0, print(loss)
#     auc = roc_auc_score(rel.detach().numpy(), pre.detach().numpy())
#     aupr = average_precision_score(rel.detach().numpy(), pre.detach().numpy())
#     #     out = model(train_data.x_dict,train_data.edge_index_dict)
#     #     pre=out[train_data['regulate'].edge_label_index[0],train_data['regulate'].edge_label_index[1]]
#     #     rel=train_data['regulate'].edge_label
#     #     loss = F.binary_cross_entropy_with_logits(pre,rel)
#     #     pre=torch.sigmoid(pre)
#     #     assert torch.isnan(loss).sum() == 0, print(loss)
#     #     auc=roc_auc_score(rel.detach().numpy(), pre.detach().numpy())
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     return (loss.item(), auc, aupr)


# def evaluateAll(model, dat, lossF):
#     model.eval()
#     valloss = 0
#     valauc = 0
#     with torch.no_grad():
#         out, _, _ = model(dat.seq_dict, dat.sim_dict, dat.edge_index_dict, dat['regulate'].edge_label_index)
#         out = out.view(-1)
#         rel = dat['regulate'].edge_label
#         # loss = F.binary_cross_entropy_with_logits(out,rel)
#         loss = lossF(out, rel)
#         pre = torch.sigmoid(out)
#         #         out = model(dat.x_dict,dat.edge_index_dict)
#         #         pre=out[dat['regulate'].edge_label_index[0],dat['regulate'].edge_label_index[1]]
#         #         rel=dat['regulate'].edge_label
#         #         loss = F.binary_cross_entropy_with_logits(pre,rel)
#         #         pre=torch.sigmoid(pre)
#         auc = roc_auc_score(rel.detach().numpy(), pre.detach().numpy())
#         aupr = average_precision_score(rel.detach().numpy(), pre.detach().numpy())
#     return (loss.item(), auc, aupr)

def trainEpoch(model,trainloader,scaler,lossF,device):
    model.train()
    trainloss=0
    auc=0
    aupr=0
    #acc=0
    for step,dat in enumerate(trainloader):
        xm1,xg1,xm2,xg2,mid,gid,rel=dat
        xm1,xg1,xm2,xg2,mid,gid,rel=xm1.to(device),xg1.to(device),xm2.to(device),xg2.to(device),\
                                    mid.to(device),gid.to(device),rel.to(device)
        
        optimizer.zero_grad()
        # 训练模型
        #with autocast():
        #out = model(dat.x_dict,dat.seq_dict, dat.edge_index_dict)
        out,_,_ = model(xm1, xg1,xm2,xg2)
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
#         if (step % 100 == 0):
#             train_loss = (trainloss / (step+1))
#             auc_batch = (auc / (step+1))
#             aupr_batch = (aupr / (step+1))
#             #train_loss.append(trainloss)
#             print('Batch:',step,train_loss,auc_batch,aupr_batch)
    return (trainloss/(step+1),auc/(step+1),aupr/(step+1))

def evaluate(model,valloader,lossF,device):
    model.eval()
    valloss=0
    valauc=0
    valaupr=0
    with torch.no_grad():
        for step,dat in enumerate(valloader):
            xm1,xg1,xm2,xg2,mid,gid,rel=dat
            xm1,xg1,xm2,xg2,mid,gid,rel=xm1.to(device),xg1.to(device),xm2.to(device),xg2.to(device),\
                                        mid.to(device),gid.to(device),rel.to(device)

            optimizer.zero_grad()
            # 训练模型
            #with autocast():
            #out = model(dat.x_dict,dat.seq_dict, dat.edge_index_dict)
            out,_,_ = model(xm1, xg1,xm2,xg2)
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
#             if (step % 100 == 0):
#                 val_loss = (valloss / (step+1))
#                 #val_loss.append(valloss)
#                 val_auc=(valauc/(step+1))
#                 val_aupr=(valaupr/(step+1))
#                 print('Batch:',step,val_loss,val_auc,val_aupr)
    return (valloss/(step+1)),(valauc/(step+1)),(valaupr/(step+1))


class EarlyStopping():
    def __init__(self, tolerance=50, min_delta=0.1):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_best_auc, val_auc):
        if val_auc < val_best_auc:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


def transfer_model(pretrained_file, model):
    pretrained_dict = torch.load(pretrained_file, map_location=torch.device('cpu'))  # get pretrained dict
    model_dict = model.state_dict()  # get model dict
    pretrained_dict = transfer_state_dict(pretrained_dict, model_dict)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def transfer_state_dict(pretrained_dict, model_dict):
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict

# class seqData(Dataset):
#     def __init__(self,xm,xg,label,edgeidx):
#         self.source= xm
#         self.target=xg
#         self.edge=edgeidx
#         self.label = label
#         self.length = len(self.label)
#     def __getitem__(self, index):
#         s=self.edge[0][index]
#         t=self.edge[1][index]
#         return self.source[s], self.target[t],s,t,self.label[index]
#     def __len__(self):
#         return self.length

class seqData(Dataset):
    def __init__(self,xm1,xg1,xm2,xg2,label,edgeidx):
        self.source1= xm1
        self.target1=xg1
        self.source2= xm2
        self.target2=xg2
        self.edge=edgeidx
        self.label = label
        self.length = len(self.label)
    def __getitem__(self, index):
        s=self.edge[0][index]
        t=self.edge[1][index]
        return self.source1[s], self.target1[t],self.source2[s], self.target2[t],s,t,self.label[index]
    def __len__(self):
        return self.length
    
if __name__ == '__main__':

    torch.cuda.set_device(1)
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments"
    #PYTORCH_CUDA_ALLOC_CONF=expandable_segments
    set_seed(2022)
    with open('dataSplit_negall_usingmiRNAanchor.pkl','rb') as f:  # Python 3: open(..., 'rb')
        _,_,_,_,trainId,trainLabel,testId,testLabel,valId,valLabel,_,m,g = pickle.load(f)
    
    with open('./results/TranCNN_mgEmbedding_usingmiRNAanchor.pkl',
              'rb') as f:  # Python 3: open(..., 'rb')
        xm1, xg1 = pickle.load(f)

    with open('./results/HGTfull_mgEmbedding_usingmiRNAanchor.pkl',
              'rb') as f:  # Python 3: open(..., 'rb')
        xm2, xg2 = pickle.load(f)
    
    train_data=seqData(xm1,xg1,xm2,xg2,trainLabel,trainId)
    val_data=seqData(xm1,xg1,xm2,xg2,valLabel,valId)
    #test_data=seqData(xm1,xm2,xg1,xg2,testLabel,testId)
    
    train_loader = DataLoader(dataset=train_data, batch_size=1024*2, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=1024*2, shuffle=True)
    #test_loader = DataLoader(dataset=test_data, batch_size=256, shuffle=True)
    
    model = HerGraph(dropout_rate=0.3)
    print(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device='cuda:1'
    model.to(device)

    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            torch.nn.init.kaiming_normal_(m.weight, mode = 'fan_in')

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=0.000001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',factor=0.5,verbose=1,min_lr=0.000001,patience=20)
    #lossF=nn.BCEWithLogitsLoss()
    #lossF=BCEFocalLoss(gamma=2, alpha=0.35, reduction='elementwise_mean')
    lossF=FocalLoss(alpha=0.65, gamma=2,reduce='mean')
    early_stopping = EarlyStopping(tolerance=200, min_delta=0.15)

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
        train_loss, train_auc, train_aupr = trainEpoch(model,train_loader,scaler,lossF,device)
        trainloss.append(train_loss)
        print('train_loss:',train_loss)
        #time_elapsed = time.time() - since
        #print('Training in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        #since1 = time.time()
        val_loss, val_auc, val_aupr = evaluate(model,val_loader,lossF,device)
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
            torch.save(model, './miRGraph_stepbystep/Modelbest_gpu_pre_0.001_usingmiRNAanchor.pt')
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
        #     torch.save(model, 'Modelbest.pt')
        # #    test_auc= tmp_test_auc
        # #scheduler.step()
        # time_elapsed = time.time() - since
        # log = 'Epoch: {:03d}, Epoch complete in {:.0f}m {:.0f}s, trainLoss: {:.4f}, Valloss: {:.4f}, Trainauc: {:.4f}, Valauc: {:.4f}, Valbestauc: {:.4f},Trainaupr: {:.4f}, Valaupr: {:.4f}, Valbestaupr: {:.4f}'
        # print(log.format(epoch, time_elapsed // 60, time_elapsed % 60,train_loss, val_loss,train_auc,val_auc,best_val_auc,train_aupr,val_aupr,best_val_aupr))

        # early_stopping(train_auc, val_auc)
        # if early_stopping.early_stop:
        #     print("We are at epoch:", epoch)
        #     break

    with open('./miRGraph_stepbystep/ModelResult_gpu_pre_0.001_usingmiRNAanchor.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([trainloss, trainauc,trainaupr,valloss,valauc,valaupr,lrchange], f)

    torch.save(model, './miRGraph_stepbystep/Model_gpu_pre_0.001_usingmiRNAanchor.pt')