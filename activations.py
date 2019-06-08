# Hardsigmoid activation function - used for ConvLSTM
class Hardsigmoid(nn.Module):
    def __init__(self):
        super(Hardsigmoid,self).__init__()
        self.hardtanh = nn.Hardtanh(-2.5,2.5)
    def forward(self,X):
        return (self.hardtanh(X) + 2.5) / 5.0

# SatLU activation function
class SatLU(nn.Module):
    """
    Ensures maximum output value <= pixel_max
    Uses either hardtanh or logsigmoid
    """
    def __init__(self,act,pixel_max):
        super(SatLU,self).__init__()
        self.act = act
        self.pixel_max = pixel_max
        if act == 'hardtanh':
            lower = 0 # Lower bound always 0
            upper = pixel_max
            self.activation = nn.Hardtanh(0,pixel_max)
        elif act == 'logsigmoid':
            self.activation = nn.LogSigmoid()
    def forward(self,input):
        if self.act == 'hardtanh':
            return self.activation(input)
        elif self.act == 'logsigmoid':
            return self.activation(input) + torch.tensor(self.pixel_max)
