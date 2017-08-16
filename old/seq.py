D_in = 5
H = 4
D_out = 3
mdl = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

D_in = 5
H = 4
D_out = 3
mdl = torch.nn.Sequential(
    torch.nn.Linear(D_in, H, bias=False),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out, bias=False)
)
