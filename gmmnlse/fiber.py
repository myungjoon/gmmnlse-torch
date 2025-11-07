class Fiber:
    def __init__(self, wvl0=1030e-9, S=None, gamma=None, betas=None, L=1., n2=0, fr=0., hrw=None):
        self.wvl0 = wvl0
        self.L = L
        self.n2 = n2
        self.S = S
        if gamma is not None:
            self.gamma = gamma
        if betas is not None:
            self.betas = betas
        self.fr = fr


        if hrw is not None:
            self.hrw = hrw
        else:
            self.hrw = 0.