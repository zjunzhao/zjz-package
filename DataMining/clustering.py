import numpy as np
import torch

class KMeans:
    
    def __init__(self, opt_mode, tot_step=100, batch_size=128, lr=1e-3):
        assert opt_mode in ['gd', 'sgd', 'adam']
        self.opt_mode = opt_mode
        self.tot_step = tot_step
        if self.opt_mode!='gd':
            self.batch_size = batch_size
            self.lr = lr
        else:
            self.lr = 1.0
    
    def train(self, X, k):
        X = torch.tensor(X)
        n, d = X.size()
        C = X[np.random.choice(n, k, False)]
        self.opt_stats = self.opt_stats_init(k, d)
        for _ in range(self.tot_step):
            if self.opt_mode=='gd':
                X0 = X.clone()
            else:
                X0 = X[np.random.choice(n, self.batch_size, False)].clone()
            C0 = C.permute(1, 0).contiguous()
            Dist0 = (X0**2).sum(dim=1, keepdim=True)+(C0**2).sum(dim=0, keepdim=True)-2*torch.mm(X0, C0)
            _, y0 = Dist0.min(dim=1)
            delta = torch.zeros_like(C)
            for i in range(k):
                delta[i] = X0[y0==i].mean(dim=0)-C[i]
            C = C+self.lr*self.calc_direction(delta)
        return C
    
    def opt_stats_init(self, k, d):
        ret = {}
        if self.opt_mode=='adam':
            ret['firstordermoment'] = torch.zeros(k, d)
            ret['firstorderweight'] = 0
            ret['secondordermoment'] = torch.zeros(k, d)
            ret['secondorderweight'] = 0
        return ret
    
    def calc_direction(self, delta):
        if self.opt_mode in ['gd', 'sgd']:
            return delta
        if self.opt_mode=='adam':
            m1, m2, eps = 1e-3, 1e-3, 1e-8
            self.opt_stats['firstordermoment'] = (1-m1)*self.opt_stats['firstordermoment']+m1*delta
            self.opt_stats['firstorderweight'] = (1-m1)*self.opt_stats['firstorderweight']+m1
            self.opt_stats['secondordermoment'] = (1-m2)*self.opt_stats['secondordermoment']+m2*(delta**2)
            self.opt_stats['secondorderweight'] = (1-m2)*self.opt_stats['secondorderweight']+m2
            return self.opt_stats['firstordermoment']/(self.opt_stats['secondordermoment'].sqrt()+eps)