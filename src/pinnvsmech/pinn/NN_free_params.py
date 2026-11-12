from torch import nn

class NN(nn.Module):
    def __init__(self, dimensions, activation):
        super(NN, self).__init__()
        layers = []
        for idx, _ in enumerate(dimensions[:-1]):
            layers += [nn.Linear(dimensions[idx], dimensions[idx+1])]
            layers += [activation()]
        layers = layers[:-1]
        layers += [nn.Sigmoid()]
        self.stack = nn.Sequential(*layers)
        
        
    def forward(self, x):
        out = self.stack(x)
        return out



# class simple_NN(nn.Module):
#     def __init__(self):
#         activation = nn.Tanh
#         dimensions = [3, 16, 32, 16, 3]
#         super(simple_NN, self).__init__(dimensions, activation)


# class simple_NN_5_100(NN):
#     def __init__(self):
#         activation = nn.Tanh
#         dimensions = [3, 100, 100, 100, 100, 100, 3]
#         super(simple_NN_5_100, self).__init__(dimensions, activation)


