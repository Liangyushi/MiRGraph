# MiRGraph: A hybrid deep learning approach to identify microRNA-target interactions by integrating heterogeneous regulatory network and genomic sequences
MicroRNAs (miRNAs) mediates gene expression regulation by targeting specific messenger RNAs (mRNAs) in the cytoplasm. They can function as both tumor suppressors and oncogenes depending on the specific miRNA and its target genes. Detecting miRNA-target interactions (MTIs) is critical for unraveling the complex mechanisms of gene regulation and promising towards RNA therapy for cancer. There is currently a lack of MTIs prediction methods that simultaneously perform feature learning from heterogeneous gene regulatory network (GRN) and genomic sequences. To improve the prediction performance of MTIs, we present a novel transformer-based multi-view feature learning method – MiRGraph, which consists of two main modules for learning the sequence-based and GRN-based feature embedding. For the former, we utilize the mature miRNA sequences and the complete 3'UTR sequence of the target mRNAs to encode sequence features using a hybrid transformer and convolutional neural network (CNN) (TransCNN) architecture. For the latter, we utilize a heterogeneous graph transformer (HGT) module to extract the relational and structural information from the GRN consisting of miRNA-miRNA, gene-gene and miRNA-target interactions. The TransCNN and HGT modules can be learned end-to-end to predict experimentally validated MTIs from MiRTarBase. MiRGraph outperforms existing methods in not only recapitulating the true MTIs but also in predicting strength of the MTIs based on the in-vitro measurements of miRNA transfections. In a case study on breast cancer, we identified plausible target genes of an oncomir.

The proposed framework MiRGraph mainly consists of three parts (Fig. 1): a. Heterogeneous information network (HIN) construction; b. Feature Encoder; c. MTI Predictor. Briefly, we first built a heterogeneous information network (HIN), and obtain the initial network and sequence one-hot encoding of miRNAs and genes (Fig. 1a). They are fed into feature encoder including two modules (i.e. HGT and TransCNN) to learn network and sequence representations separately (Fig. 1b). Finally, we feed the concatenated representation into the MTI predictor (Fig. 1c). Details of each step are described below.

<img width="4552" height="4375" alt="fig31" src="https://github.com/user-attachments/assets/dffb6bb6-06e8-4f1e-965b-4bfa5f83e896" />

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

# Citation
P. Liu, Y. Liu, J. Luo and Y. Li, "MiRGraph: A hybrid deep learning approach to identify microRNA-target interactions by integrating heterogeneous regulatory network and genomic sequences," 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), Lisbon, Portugal, 2024, pp. 1028-1035, doi: 10.1109/BIBM62325.2024.10822436.
