import torch.nn as nn


class MyEnsemble(nn.Module):
    def __init__(self, modelA,classifier ):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA.encoder
        self.classifier = classifier

    def forward(self, x):
        x1 = self.modelA(x)
        x = self.classifier(x1)
        return x
    
class MyUQEnsemble(nn.Module):
    def __init__(self, modelA, classifier, ugraft_probing=False):
        super(MyUQEnsemble, self).__init__()
        self.ugraft_probing = ugraft_probing

        self.modelA = modelA
        self.classifier = classifier

    def forward(self, x):

        if self.ugraft_probing:
            x1, variance = self.modelA(x)

        else:
            x1 = self.modelA.encoder(x)
            _, variance = self.modelA(x)

        x = self.classifier(x1)
        return x, variance