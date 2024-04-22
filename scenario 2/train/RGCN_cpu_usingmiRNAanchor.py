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
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
#from torchsummary import summary
import random
from torch_geometric.nn import RGCNConv
import copy

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(2022)


class RGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,num_relations,node_types,init_sizes,num_nodes):
        super(RGCN, self).__init__()
        self.num_nodes=num_nodes
        self.conv1 = RGCNConv(in_channels, hidden_channels,num_relations, num_bases=10)
        self.conv2 = RGCNConv(hidden_channels, out_channels,num_relations, num_bases=10)
        self.lins = torch.nn.ModuleList()
        for i in range(len(node_types)):
            lin = nn.Linear(init_sizes[i], in_channels)
            self.lins.append(lin)
#         self.relu = nn.relu
#         self.lin1 = Linear(out_channels, out_channels)
#         self.lin2 = Linear(out_channels, out_channels)
#         self.dropout = dropout   
    def trans_dimensions(self, g):
        data = copy.deepcopy(g)
        for node_type, lin in zip(node_types, self.lins):
            data[node_type].sim = lin(data[node_type].sim)
        return data
    
    def forward(self, data):
        data = self.trans_dimensions(data)
        homogeneous_data = data.to_homogeneous()
        #print(homogeneous_data)
        edge_index, edge_type = homogeneous_data.edge_index, homogeneous_data.edge_type
        x = self.conv1(homogeneous_data.sim, edge_index, edge_type)
        x = self.conv2(x, edge_index, edge_type)
        xm = x[:self.num_nodes]
        xg = x[self.num_nodes:]
#         xm = self.relu(self.lin1(xm))
#         xg = self.relu(self.lin1(xg))
        return xm,xg

# class RGCN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels,num_relations,node_types,init_sizes):
#         super(RGCN, self).__init__()
#         #self.num_nodes=num_nodes
#         self.conv1 = RGCNConv(in_channels, hidden_channels,num_relations, num_bases=10)
#         self.conv2 = RGCNConv(hidden_channels, out_channels,num_relations, num_bases=10)
#         self.lins = torch.nn.ModuleList()
#         for i in range(len(node_types)):
#             lin = nn.Linear(init_sizes[i], in_channels)
#             self.lins.append(lin)
#     #         self.relu = nn.relu
#     #         self.lin1 = Linear(out_channels, out_channels)
#     #         self.lin2 = Linear(out_channels, out_channels)
#     #         self.dropout = dropout   
#     def trans_dimensions(self, data):
#         #data = copy.deepcopy(g)
#         for node_type, lin in zip(node_types, self.lins):
#             data[node_type].x = lin(data[node_type].x)
#         return data

#     def forward(self, data):
#         num_nodes = data['miRNA'].num_nodes
#         data = self.trans_dimensions(data)
#         #homogeneous_data = data.to_homogeneous()
#         data = data.to_homogeneous()
#         #print(homogeneous_data)
#         edge_index, edge_type = data.edge_index, data.edge_type
#         x = self.conv1(data.x, edge_index, edge_type)
#         x = self.conv2(x, edge_index, edge_type)
#         xm = x[:num_nodes]
#         xg = x[num_nodes:]
#         #         xm = self.relu(self.lin1(xm))
#         #         xg = self.relu(self.lin1(xg))
#         return xm,xg
    
class MLPBilPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout_rate=0.3):
        super(MLPBilPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.bilin = torch.nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.relu = nn.GELU()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        self.bilin.reset_parameters()

    def forward(self, x_i, x_j):
        for lin in self.lins:
            x_i, x_j = lin(x_i), lin(x_j)
            x_i, x_j = self.relu(x_i), self.relu(x_j)
        x = torch.sum(self.bilin(x_i) * x_j, dim=-1)
        return x

class RGCNpre(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,num_relations,node_types,init_sizes,num_nodes,dropout_rate=0.1):
        super(RGCNpre, self).__init__()
        self.rgcn=RGCN(in_channels, hidden_channels,out_channels,num_relations,node_types,init_sizes,num_nodes)
        #self.bilstm=BiLSTM(input_size, hidden_size, num_layers, output_size,dropout_rate)

        #self.embg = Linear(output_size+out_channels,128,False)
        self.pre=MLPBilPredictor(128, 128, 1, 0.3)
        #self.relu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
    def encoder(self,data,seqG):
        
        xm,xg=self.rgcn(data)
#         print(xm.size(),xg1.size())
#         xg2=self.bilstm(seqG)
#         print(xg2.size())
#         xg=self.embg(torch.cat([xg1, xg2], dim=1))
#         print(xg.size())
#         xm = self.dropout(xm)
#         xg = self.dropout(xg)
        #print(xm.size())
        #print(xg.size())
        return xm,xg
    def decoderMLP(self,xm,xg,edge):
        xm=xm[edge[0]]
        xg=xg[edge[1]]
        #s=self.pre(xm,xg)
        s=torch.sum(xm * xg, dim=-1)
        return s
    def forward(self,data,seqG,label_edge):
        xm,xg=self.encoder(data,seqG)
        s=self.decoderMLP(xm,xg,label_edge)
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

def trainAll(model,train_data,seqG,lossF):
    model.train() 
    out = model(train_data,seqG,train_data['regulate'].edge_label_index)
    out=out.view(-1)
    rel=train_data['regulate'].edge_label
    #loss = F.binary_cross_entropy_with_logits(out,rel)
    loss=lossF(out,rel)
    pre=torch.sigmoid(out)
    assert torch.isnan(loss).sum() == 0, print(loss)
    auc=roc_auc_score(rel.detach().numpy(), pre.detach().numpy())
    aupr= average_precision_score(rel.detach().numpy(), pre.detach().numpy())
    #     out = model(train_data.x_dict,train_data.edge_index_dict)
    #     pre=out[train_data['regulate'].edge_label_index[0],train_data['regulate'].edge_label_index[1]]
    #     rel=train_data['regulate'].edge_label
    #     loss = F.binary_cross_entropy_with_logits(pre,rel)
    #     pre=torch.sigmoid(pre)
    #     assert torch.isnan(loss).sum() == 0, print(loss)
    #     auc=roc_auc_score(rel.detach().numpy(), pre.detach().numpy())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return (loss.item(),auc,aupr)

def evaluateAll(model,dat,seqG,lossF):
    model.eval()
    valloss=0
    valauc=0
    with torch.no_grad():
        out = model(dat,seqG,dat['regulate'].edge_label_index)
        out=out.view(-1)
        rel=dat['regulate'].edge_label
        #loss = F.binary_cross_entropy_with_logits(out,rel)
        loss=lossF(out,rel)
        pre=torch.sigmoid(out)
        #         out = model(dat.x_dict,dat.edge_index_dict)
        #         pre=out[dat['regulate'].edge_label_index[0],dat['regulate'].edge_label_index[1]]
        #         rel=dat['regulate'].edge_label
        #         loss = F.binary_cross_entropy_with_logits(pre,rel)
        #         pre=torch.sigmoid(pre)
        auc=roc_auc_score(rel.detach().numpy(), pre.detach().numpy())
        aupr= average_precision_score(rel.detach().numpy(), pre.detach().numpy())
    return (loss.item(),auc,aupr)

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

def transfer_model(pretrained_file, model):
    pretrained_dict = torch.load(pretrained_file,map_location=torch.device('cpu'))  # get pretrained dict
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

if __name__ == '__main__':

    with open('dataCombine_negall_usingmiRNAanchor.pkl','rb') as f:  # Python 3: open(..., 'rb')
        train_data,val_data,test_data,m,g = pickle.load(f)

    train_data ['miRNA'].sim = train_data ['miRNA'].mm + train_data ['miRNA'].x
    val_data ['miRNA'].sim = train_data ['miRNA'].mm + train_data ['miRNA'].x
    test_data ['miRNA'].sim = train_data ['miRNA'].mm + train_data ['miRNA'].x
    train_data ['gene'].sim = train_data ['gene'].gg + train_data ['gene'].x
    val_data ['gene'].sim = train_data ['gene'].gg + train_data ['gene'].x
    test_data ['gene'].sim = train_data ['gene'].gg + train_data ['gene'].x
    
    node_types, edge_types = train_data.metadata()
    num_relations = len(edge_types)
    init_sizes = [train_data[x].x.shape[1] for x in node_types]
    num_nodes = train_data['miRNA'].x.shape[0]
    
    print(num_nodes)
    seqG=np.load('geneseqkmer.npy')
    seqG=torch.from_numpy(seqG)
    seqG=seqG.float()
    #print(seqG.size())
    
    model = RGCNpre(in_channels=1024, hidden_channels=256, out_channels=128,\
                        num_relations=num_relations,node_types=node_types,init_sizes=init_sizes,num_nodes=num_nodes,dropout_rate=0.3)

    print(model)

    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            torch.nn.init.xavier_normal_(m.weight)
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
        train_loss,train_auc,train_aupr = trainAll(model,train_data,seqG,lossF)
        trainloss.append(train_loss)
        print('train_loss:',train_loss)
        #time_elapsed = time.time() - since
        #print('Training in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        #since1 = time.time()
        val_loss,val_auc,val_aupr = evaluateAll(model,val_data,seqG,lossF)
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
            torch.save(model, './RGCN/RGCNbest_cpu_directDot_usingmiRNAanchor.pt')
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

        # time_elapsed = time.time() - since
        # log = 'Epoch: {:03d}, Epoch complete in {:.0f}m {:.0f}s, trainLoss: {:.4f}, Valloss: {:.4f}, Trainauc: {:.4f}, Valauc: {:.4f}, Valbestauc: {:.4f},Trainaupr: {:.4f}, Valaupr: {:.4f}, Valbestaupr: {:.4f}'
        # print(log.format(epoch, time_elapsed // 60, time_elapsed % 60,train_loss, val_loss,train_auc,val_auc,best_val_auc,train_aupr,val_aupr,best_val_aupr))

    with open('./RGCN/RGCNResult_cpu_directDot_usingmiRNAanchor.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([trainloss, trainauc,trainaupr,valloss,valauc,valaupr,lrchange], f)

    torch.save(model, './RGCN/RGCN_cpu_directDot_usingmiRNAanchor.pt')