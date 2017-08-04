def w_init():
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def get_example_mdl():
    mdl = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    )
    return mdl

mdl = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

def w_init_randn(data,mu=0.0,stddev=1.0):
    #l.weight.data.normal_(mu,stddev)
    data.normal_(mu,stddev)
    print(data)

wf = lambda l: l.weight.data.normal_(0,1)
wf = lambda l: l.weight.data.fill_(0)

def w_init_fill(data,val=0.0):
    '''

    '''
    #l.weight.data.normal_(mu,stddev)
    data.fill_(val)
    print(data)

def _old_forward(self, x):
    a_l1 = self.linear1(x)**2 # [M,H^(1)] = [M,D]x[D,H^(1)]
    a_l2 = a_l1.mm(W_l2)**2 # [M,H^(2)] = [M,H^(1)]x[H^(1),H^(2)]
    y_pred = a_l2.mm(W_out) # [M,1] = [M,H^(2)]x[M^(2),1]
    return y_pred
