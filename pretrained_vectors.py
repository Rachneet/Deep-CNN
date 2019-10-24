import numpy as np
import pickle
import bcolz
import torch.nn as nn
import torch

path = "~/dpcnn"
# words = []
# word2idx = {}
# idx = 0
# vectors = bcolz.carray(np.zeros(1), rootdir=f'{path}/fastext_vec.dat', mode='w')
#
# with open(f"{path}/crawl-300d-2M.vec", "rb") as fastext:
#     # next(fastext)
#     fastext.readline()
#     for l in fastext:
#         line = l.decode().split()
#         word = line[0]
#         words.append(word)
#         word2idx[word] = idx
#         idx += 1
#         vect = np.array(line[1:]).astype(np.float)
#         vectors.append(vect)
#         # print(vectors)
#
# vectors = bcolz.carray(vectors[1:].reshape(1999995, 300), rootdir=f"{path}/fastext_vec.dat", mode='w')
# vectors.flush()
# pickle.dump(words, open(f'{path}/ft_words.pkl', 'wb'))
# pickle.dump(word2idx, open(f'{path}/ft_w2idx.pkl', 'wb'))

#load vecs and make dict
# vectors = bcolz.open(f'{path}/fastext_vec.dat')[:]
# words = pickle.load(open(f'{path}/ft_words.pkl', 'rb'))
# word2idx = pickle.load(open(f'{path}/ft_w2idx.pkl', 'rb'))
#
# fastext = {w:vectors[word2idx[w]] for w in words}
#
# print(fastext['the'])

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape[0], weights_matrix.shape[1]
    print(num_embeddings,embedding_dim)
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


