import pickle

import torch


# 示例的输入字典列表
list_of_dicts = []

for i in range(0,8):
    with open('df_query3_swinL_ddp_384-48-'+str(i)+'.pkl', 'rb') as f:
        list_of_dicts.append(pickle.load(f))

# 创建一个新的字典来保存平均值
averaged_dict = {}

# 遍历每个键
for key in list_of_dicts[0].keys():
    # 获取当前键的值列表
    values = [torch.tensor(d[key])  for d in list_of_dicts]
    # 计算平均值
    averaged_value = torch.stack(values).mean(dim=0)
    # 将平均值保存到新的字典中
    averaged_dict[key] = averaged_value.tolist()

with open("df_query3_swinL_ddp_384-48-tta.pkl", "wb") as f:
    pickle.dump(averaged_dict, f)

list_of_dicts = []

for i in range(0,8):
    with open('df_gallery3_swinL_ddp_384-48-'+str(i)+'.pkl', 'rb') as f:
        list_of_dicts.append(pickle.load(f))

# 创建一个新的字典来保存平均值
averaged_dict = {}

# 遍历每个键
for key in list_of_dicts[0].keys():
    # 获取当前键的值列表
    values = [torch.tensor(d[key])  for d in list_of_dicts]
    # 计算平均值
    averaged_value = torch.stack(values).mean(dim=0)
    # 将平均值保存到新的字典中
    averaged_dict[key] = averaged_value.tolist()

with open("df_gallery3_swinL_ddp_384-48-tta.pkl", "wb") as f:
    pickle.dump(averaged_dict, f)
