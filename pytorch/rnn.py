import torch
import torch.nn as nn

rnn_type="LSTM"
rnn = getattr(nn, rnn_type)
print("rnn:", rnn)

emb_dim=10
hidden_size=6
num_layers=1

rnn_layer = getattr(nn, rnn_type)(input_size=emb_dim, hidden_size=hidden_size, num_layers=num_layers)
print("rnn_layer:", rnn_layer)

seq_len=5
batch_size=4
emb = torch.rand([batch_size, seq_len, emb_dim])

print("Input: emb.shape:", emb.shape)
print("Input: emb:", emb)
hidden_list, hidden_final = rnn_layer(emb)
print("Output: hidden_list.shape:", hidden_list.shape)
print("Output: len(hidden_final):", len(hidden_final))
print("Output: hidden_final[0].shape:", hidden_final[0].shape)
print("Output: hidden_final[1].shape:", hidden_final[1].shape)

print("Output: hidden_list:", hidden_list)
print("Output: hidden_final:", hidden_final)
