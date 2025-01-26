import pandas as pd
import numpy as np
from linear_feature import get_low_feature

#association = pd.read_excel("./data1/lncRNADisease-lncRNA-disease associations matrix.xls", header=0, index_col=0)
association = pd.read_excel("./data2/MNDR-lncRNA-disease associations matrix.xls", header=0, index_col=0)
#linear feature
feature_MFl, feature_MFd = get_low_feature(64, 0.01, pow(10, -4), association.values)


# #nonlinear feature
feature_nl = pd.read_csv("./data2/m64_feature.csv", header=None, index_col=None).values
feature_nd = pd.read_csv("./data2/d64_feature.csv",header=None, index_col=None).values
#
# #feature merger
rna_feature = np.hstack((feature_MFl,feature_nl))
disease_feature = np.hstack((feature_MFd,feature_nd))

lncrna = pd.DataFrame(rna_feature)
lncrna.index = association.index.tolist()
lncrna.to_csv('./data2/rna_feature.csv')
disease = pd.DataFrame(disease_feature)
disease.index = association.columns.to_list()
disease.to_csv('./data2/disease_feature.csv')




all_associations = pd.read_csv('./data2' + '/pair.txt', sep=' ', names=['r', 'd', 'label'])

#label = pd.read_excel('./data1/lncRNADisease-lncRNA-disease associations matrix.xls',header=0,index_col=0)
label = pd.read_excel('./data2/MNDR-lncRNA-disease associations matrix.xls',header=0,index_col=0)
label.to_csv("./data2/label.csv",header=None,index=None)


dataset = []

for i in range(int(all_associations.shape[0])):
    r = all_associations.iloc[i, 0]-1
    c = all_associations.iloc[i, 1]-1
    label = all_associations.iloc[i, 2]
    # dataset.append(np.hstack((feature_MFl[r], feature_MFd[c], label)))
    # dataset.append(np.hstack((feature_nl[r], feature_nd[c], label)))
    dataset.append(np.hstack((rna_feature[r], disease_feature[c], label)))
all_dataset = pd.DataFrame(dataset)

all_dataset.to_csv("./data2/data64.csv",header=None,index=None)

print("Fnished!")