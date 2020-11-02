import math
import torch

def LpNorm(x, p=2):
    return (x**p).sum()**(1/p)

class SVD:
    
    def __init__(self):
        pass
    
    def decompose(self, M, r, totstep=1000):
        M = torch.tensor(M).float()
        MT = M.permute(1, 0).contiguous()
        m, n = M.size()
        if m<n:
            MM = torch.mm(M, MT)
        else:
            MM = torch.mm(MT, M)
        U, Sigma, V = [], [], []
        for _ in range(r):
            x = torch.rand(min(m, n), 1).float()
            for _ in range(totstep):
                x = torch.mm(MM, x)
                x = x/LpNorm(x)
            la = LpNorm(torch.mm(MM, x))
            MM = MM-la*torch.mm(x, x.permute(1, 0).contiguous())
            s = math.sqrt(la)
            if m<n:
                U.append(x)
                Sigma.append(s)
                V.append(torch.mm(MT, x)/s)
            else:
                V.append(x)
                Sigma.append(s)
                U.append(torch.mm(M, x)/s)
        U = torch.cat(U, dim=1).numpy()
        Sigma = torch.diag(torch.tensor(Sigma)).numpy()
        V = torch.cat(V, dim=1).numpy()
        return U, Sigma, V
