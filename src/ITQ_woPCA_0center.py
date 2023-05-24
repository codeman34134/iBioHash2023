import pickle

import pandas as pd
import torch
import numpy as np



def train(
        train_data,
        code_length,
        max_iter,
        device
):
    """
    Training model.
    Args
        train_data(torch.Tensor): Training data.
        query_data(torch.Tensor): Query data.
        query_targets(torch.Tensor): Query targets.
        retrieval_data(torch.Tensor): Retrieval data.
        retrieval_targets(torch.Tensor): Retrieval targets.
        code_length(int): Hash code length.
        max_iter(int): Number of iterations.
        device(torch.device): GPU or CPU.
        topk(int): Calculate top k data points map.
    Returns
        checkpoint(dict): Checkpoint.
    """
    # Initialization
    R = torch.randn(code_length, code_length).to(device)
    [U, _, _] = torch.svd(R)
    R = U[:, :code_length]

    # PCA
    V = train_data.to(device)

    # Training
    for i in range(max_iter):
        V_tilde = V @ R
        B = V_tilde.sign()
        [U, _, VT] = torch.svd(B.t() @ V)
        R = (VT.t() @ U.t())


    # Save checkpoint
    checkpoint = {
        'rotation_matrix': R
    }

    return checkpoint


def generate_code(data, R):
    """
    Generate hashing code.
    Args
        data(torch.Tensor): Data.
        code_length(int): Hashing code length.
        R(torch.Tensor): Rotration matrix.
        pca(callable): PCA function.
    Returns
        pca_data(torch.Tensor): PCA data.
    """
    return (data.to(R.device) @ R).sign()

with open('/home/hdd/ct_RecallatK_surrogate/src/df_query0_swinL_ddp_192-48.pkl', 'rb') as f:
    df_query_ = pickle.load(f)
with open('/home/hdd/ct_RecallatK_surrogate/src/df_gallery0_swinL_ddp_192-48.pkl', 'rb') as f:
    df_gallery_ = pickle.load(f)

df_query = torch.Tensor(list(df_query_.values()))
df_query = torch.nn.functional.normalize(df_query, p=2, dim=1)
df_gallery = torch.Tensor(list(df_gallery_.values()))
df_gallery = torch.nn.functional.normalize(df_gallery, p=2, dim=1)
train_data = torch.cat((df_query,df_gallery),dim=0)
# 0均值
train_data = train_data - torch.mean(train_data, dim=0,keepdim=True)
checkpoint = train(train_data=train_data,code_length=48,max_iter=700,device='cuda')

# with open("ITQ_ckpt.pkl", "wb") as f:
#     pickle.dump(checkpoint, f)
#
# with open('ITQ_ckpt.pkl', 'rb') as f:
#     checkpoint = pickle.load(f)

df_query_itq_tensor = generate_code(data=df_query,R=checkpoint['rotation_matrix'])
df_gallery_itq_tensor = generate_code(data=df_gallery,R=checkpoint['rotation_matrix'])

df_gallery = pd.DataFrame(columns=['image_id','hashcode'])
df_query = pd.DataFrame(columns=['image_id','hashcode'])

j=0
for i in df_query_.keys():
    output = torch.nn.functional.relu(df_query_itq_tensor[j],inplace = True).int().cpu().numpy().tolist()
    j+=1
    output = [str(x) for x in output]
    df_query = pd.concat([df_query,pd.DataFrame({"image_id":i,"hashcode":'\''+ "".join(output) + '\''},index=[0])], ignore_index=True)

j=0
for i in df_gallery_.keys():
    # if i in used_gallery:
    output = torch.nn.functional.relu(df_gallery_itq_tensor[j], inplace=True).int().cpu().numpy().tolist()
    j += 1
    output = [str(x) for x in output]
    df_gallery = pd.concat([df_gallery, pd.DataFrame({"image_id": i, "hashcode": '\'' + "".join(output) + '\''}, index=[0])],
                           ignore_index=True)
# else:
#     df_gallery = pd.concat(
#         [df_gallery, pd.DataFrame({"image_id": i, "hashcode": '\'' + "1"*48 + '\''}, index=[0])],
#         ignore_index=True)

df_query.to_csv('submit_query.csv',index=False)
df_gallery.to_csv('submit_gallery.csv',index=False)