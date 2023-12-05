import torch
from torch import nn

def init_weights(m):
  if type(m) == nn.Linear:
    torch.nn.init.uniform_(m.weight, -0.1, 0.1)
    m.bias.data.fill_(0.01)

class SelfAttention(nn.Module):
  def __init__(self, attention_size,
         batch_first=False,
         layers=1,
         dropout=.0,
         non_linearity="tanh"):
    super(SelfAttention, self).__init__()

    self.batch_first = batch_first

    if non_linearity == "relu":
      activation = nn.ReLU()
    else:
      activation = nn.Tanh()

    modules = []
    for i in range(layers - 1):
      modules.append(nn.Linear(attention_size, attention_size))
      modules.append(activation)
      modules.append(nn.Dropout(dropout))

    # last attention layer must output 1
    modules.append(nn.Linear(attention_size, 1))
    modules.append(activation)
    modules.append(nn.Dropout(dropout))

    self.attention = nn.Sequential(*modules)
    self.attention.apply(init_weights) 
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, inputs):

    ##################################################################
    # STEP 1 - perform dot product
    # of the attention vector and each hidden state
    ##################################################################

    # inputs is a 3D Tensor: batch, len, hidden_size
    # scores is a 2D Tensor: batch, len
    attention_vec = self.attention(inputs).squeeze()

    # Use optimized torch.bmm for dot product
    scores = torch.bmm(attention_vec.unsqueeze(1), inputs.transpose(1, 2)).squeeze()

    # Cache attention weights
    self.attention_weights = self.softmax(scores)

    ##################################################################
    # Step 2 - Weighted sum of hidden states, by the attention scores
    ##################################################################

    weighted = torch.mul(inputs, self.attention_weights.unsqueeze(-1).expand_as(inputs))

    # sum the hidden states
    representations = weighted.sum(1).squeeze()
    return representations