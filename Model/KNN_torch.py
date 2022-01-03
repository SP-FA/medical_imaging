import numpy as np

class KNN:
    def __init__(self, train_x, train_y, test_x, test_y, k):
        m = test_x.size(0)
        n = train_x.size(0)

        # cal Eud distance mat
        xx = (test_x**2).sum(dim=1,keepdim=True).expand(m, n)
        yy = (train_x**2).sum(dim=1, keepdim=True).expand(n, m).transpose(0, 1)
        
        dist_mat = xx + yy - 2*test_x.matmul(train_x.transpose(0, 1))
        mink_idxs = dist_mat.argsort(dim=-1)

        self.res = []
        for idxs in mink_idxs:
            # voting
            self.res.append(np.bincount(np.array([train_y[idx].item() for idx in idxs[:k]])).argmax())
        assert len(self.res) == len(test_y)

    def get_res(self):
        return self.res