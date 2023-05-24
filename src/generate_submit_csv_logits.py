import argparse
import pickle

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--query', type=str, default='submit_query.csv', help='48-bit hash code file for query set.')
parser.add_argument('--gallery', type=str, default='submit_gallery.csv', help='48-bit hash code file for gallery set.')
parser.add_argument('--submit', type=str, default='submit.csv', help='Final submit csv file.')
parser.add_argument('--k', type=int, default=20, help='Topk == 20.')
args = parser.parse_args()

k = args.k
query_code_path = args.query
gallery_code_path = args.gallery
submit_path = args.submit

with open('/home/hdd/ct_RecallatK_surrogate/src/df_query0_swinL_ddp_192-48.pkl', 'rb') as f:
    df_query = pickle.load(f)
with open('/home/hdd/ct_RecallatK_surrogate/src/df_gallery0_swinL_ddp_192-48.pkl', 'rb') as f:
    df_gallery = pickle.load(f)

query_code =torch.Tensor(list(df_query.values()))
query_code = torch.nn.functional.normalize(query_code, p=2, dim=1)
query_images = list(df_query.keys())

gallery_code = torch.Tensor(list(df_gallery.values()))
gallery_code = torch.nn.functional.normalize(gallery_code, p=2, dim=1)
gallery_images = list(df_gallery.keys())

with open(submit_path, 'w') as f:
    f.write('Id,Predicted\n')
    for i, q in enumerate(query_code):
        cos_sim = q@gallery_code.T
        _,index = torch.topk(cos_sim, k, largest=True)
        retrieval_images = [gallery_images[j][:-4] for j in index]
        f.write(query_images[i] + ',')
        f.write(' '.join(retrieval_images) + '\n')
        print(f'\r writing {i + 1}/{query_code.shape[0]}', end='')