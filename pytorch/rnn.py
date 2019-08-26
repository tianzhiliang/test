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

def example1():
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

def example2():
    lstm = nn.LSTM(emb_dim, hidden_size)  # Input dim is 3, output dim is 3
    #inputs = torch.tensor([torch.randn([1, emb_dim]) for _ in range(5)]) # make a sequence of length 5
    inputs = [torch.randn([1, emb_dim]) for _ in range(seq_len)] # make a sequence of length 5
    inputs = torch.stack(inputs)

    # initialize the hidden state.
    hidden = (torch.randn(1, 1, hidden_size), torch.randn(1, 1, hidden_size))

    print("inputs.shape:", inputs.shape)
    print("hidden.shape[0]:", hidden[0].shape)
    for i in inputs:
        out, hidden = lstm(i.view(1, 1, -1), hidden)

    #inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    inputs = inputs.view(len(inputs), 1, -1)
    hidden = (torch.randn(1, 1, hidden_size), torch.randn(1, 1, hidden_size))  # clean out hidden state
    out, hidden = lstm(inputs, hidden)
    print(out)
    print(hidden)

def example3(): # dimension wrong
    lstm = nn.LSTM(emb_dim, hidden_size)  # Input dim is 3, output dim is 3
    #inputs = torch.tensor([torch.randn([1, emb_dim]) for _ in range(5)]) # make a sequence of length 5
    inputs = [torch.randn(emb_dim) for _ in range(seq_len)] # make a sequence of length 5
    inputs = torch.stack(inputs)

    # initialize the hidden state.
    hidden = (torch.randn(1, hidden_size), torch.randn(1, hidden_size))

    print("inputs.shape:", inputs.shape)
    print("hidden.shape[0]:", hidden[0].shape)
    for i in inputs:
        out, hidden = lstm(i.view(1, -1), hidden)

    #inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    inputs = inputs.view(len(inputs), 1, -1)
    hidden = (torch.randn(1, 1, hidden_size), torch.randn(1, 1, hidden_size))  # clean out hidden state
    out, hidden = lstm(inputs, hidden)
    print(out)
    print(hidden)

#example1()
#example2()
#example3()
