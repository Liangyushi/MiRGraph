# MiRGraph
The source code of the paper “MiRGraph: learning of microRNA-mRNA interactomes from heterogeneous gene regulatory network and genomic sequences”

## Runing Environment
python 3.11.8

cuda 12.1

pytorch 2.2.1

torch_geometric 2.5.0

ryp2

## Scenario 1
### Prepreocessed Data
Input data can be obtained from this link [Input Data] (https://drive.google.com/file/d/1-oYgciNZEe-ubRLzwB9BSWLDlZR7Pz3J/view?usp=drive_link).
1. File 'dataCombine_negall.pkl' used as the input of model with network.
2. File 'dataSplit_negall.pkl' used as the input of model without network.
### Training Model
1. After configuring the environment, directly run the .py file in the **./scenario 1/train/** folder:
- running file 'HGT_BiLSTM_gpu_mlp.py' to train **HGT_BiLSTM**.
- running file 'HGT_linkloader.py' to train **HGT**.
- running file 'RGCN_BiLSTM_gpu_directDot.py' to train **MRMTI**.
- running file 'RGCN_gpu_directDot.py' to train **RGCN**.
- running file 'TransCNN.py' to train **TransCNN**.
2. For file 'miRGraph_endtoend_cpu_pre_nodj_0.0001.py', we should:
- First, running file 'HGT_linkloader.py' and 'TransCNN.py' to pretrain **HGT** and **TransCNN**.
- Then, running file 'HGT&TransCNN_embedding.ipynb' to obtain the parameters of **HGT** and **TransCNN** in **miRGraph_endtoend**.
- Finally, running file 'miRGraph_endtoend_cpu_pre_nodj_0.0001.py' to train the model **miRGraph_endtoend**.
3. For file 'miRGraph_stepbystep_gpu_0.001.py',we should:
- First, running file 'HGT_linkloader.py' and 'TransCNN.py' to pretrain **HGT** and **TransCNN**.
- Then, running file 'HGT&TransCNN_embedding.ipynb' to obtain the embedding of gene and miRNA.
- Finally, running file 'miRGraph_stepbystep_gpu_0.001.py' to train the model **miRGraph_stepbystep**.
### Testing Model
1. Directly running the .ipynb file in the **./scenario 1/test/** folder to obtain the testing results of all methods.
2. Testing results of all methods are in **./scenario 1/test/results/** folder, directly running file 'AllmethodMetric.ipynb' can obtain metrics of them.

## Scenario 2
### Prepreocessed Data
Input data can be obtained from this link [Input Data] (https://drive.google.com/file/d/1-oYgciNZEe-ubRLzwB9BSWLDlZR7Pz3J/view?usp=drive_link).
1. File 'dataCombine_negall_usingmiRNAanchor.pkl' used as the input of model with network.
2. File 'dataSplit_negall_usingmiRNAanchor.pkl' used as the input of model without network.
### Training Model
1. After configuring the environment, directly run the .py file in the **./scenario 2/train/** folder:
- running file 'HGT_BiLSTM_cpu_usingmiRNAanchor.py' to train **HGT_BiLSTM**.
- running file 'HGTfull_usingmiRNAanchor.py' to train **HGT**.
- running file 'RGCN_BiLSTM_cpu_usingmiRNAanchor.py' to train **MRMTI**.
- running file 'RGCN_cpu_usingmiRNAanchor.py' to train **RGCN**.
- running file 'TransCNN_usingmiRNAanchor.py' to train **TransCNN**.
2. For file 'miRGraph_endtoend_cpu_pre_nodj_0.0001.py', we should:
- First, running file 'HGTfull_usingmiRNAanchor.py' and 'TransCNN_usingmiRNAanchor.py' to pretrain **HGT** and **TransCNN**.
- Then, running file 'HGT&TransCNN_embedding_usingmiRNAanchor.ipynb' to obtain the parameters of **HGT** and **TransCNN** in **miRGraph_endtoend**.
- Finally, running file 'miRGraph_endtoend_cpu_pre_nodj_0.0001_usingmiRNAanchor.py' to train the model **miRGraph_endtoend**.
3. For file 'miRGraph_stepbystep_gpu_0.001.py',we should:
- First, running file 'HGTfull_usingmiRNAanchor.py' and 'TransCNN_usingmiRNAanchor.py' to pretrain **HGT** and **TransCNN**.
- Then, running file 'HGT&TransCNN_embedding_usingmiRNAanchor.ipynb' to obtain the embedding of gene and miRNA.
- Finally, running file 'miRGraph_stepbystep_gpu_0.001_usingmiRNAanchor.py' to train the model **miRGraph_stepbystep**.
### Testing Model
1. Directly running the .ipynb file in the **./scenario 2/test/** folder to obtain the testing results of all methods.
2. Testing results of all methods are in **./scenario 2/test/results/** folder, directly running file 'AllmethodMetric_usingmiRNA.ipynb' can obtain metrics of them.


