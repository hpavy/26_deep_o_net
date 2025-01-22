import torch
import torch.nn as nn
from torch.autograd import grad


def pde(
    U,
    input,
    Re,
    x_std,
    y_std,
    u_mean,
    v_mean,
    p_std,
    t_std,
    t_mean,
    u_std,
    v_std,
    w_0,
    L_adim,
    V_adim,
    ya0_mean,
    ya0_std,
):
    # je sais qu'il fonctionne bien ! Il a été vérifié
    """Calcul la pde

    Args:
        U (_type_): u,v,p calcullés par le NN
        input (_type_): l'input (x,y,t)
    """
    u = U[:, 0].reshape(-1, 1)
    v = U[:, 1].reshape(-1, 1)
    p = U[:, 2].reshape(-1, 1)

    u_X = grad(
        outputs=u,
        inputs=input,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]
    v_X = grad(
        outputs=v,
        inputs=input,
        grad_outputs=torch.ones_like(v),
        retain_graph=True,
        create_graph=True,
    )[0]
    p_X = grad(
        outputs=p,
        inputs=input,
        grad_outputs=torch.ones_like(p),
        retain_graph=True,
        create_graph=True,
    )[0]
    u_x = u_X[:, 0].reshape(-1, 1)
    u_y = u_X[:, 1].reshape(-1, 1)
    u_t = u_X[:, 2].reshape(-1, 1)
    v_x = v_X[:, 0].reshape(-1, 1)
    v_y = v_X[:, 1].reshape(-1, 1)
    v_t = v_X[:, 2].reshape(-1, 1)
    p_x = p_X[:, 0].reshape(-1, 1)
    p_y = p_X[:, 1].reshape(-1, 1)

    # Dans les prochaines lignes on peut améliorer le code (on fait des calculs inutiles)
    u_xx = grad(
        outputs=u_x, inputs=input, grad_outputs=torch.ones_like(u_x), retain_graph=True
    )[0][:, 0].reshape(-1, 1)
    u_yy = grad(
        outputs=u_y, inputs=input, grad_outputs=torch.ones_like(u_y), retain_graph=True
    )[0][:, 1].reshape(-1, 1)
    v_xx = grad(
        outputs=v_x, inputs=input, grad_outputs=torch.ones_like(v_x), retain_graph=True
    )[0][:, 0].reshape(-1, 1)
    v_yy = grad(
        outputs=v_y, inputs=input, grad_outputs=torch.ones_like(v_y), retain_graph=True
    )[0][:, 1].reshape(-1, 1)

    equ_1 = (
        (u_std / t_std) * u_t
        + (u * u_std + u_mean) * (u_std / x_std) * u_x
        + (v * v_std + v_mean) * (u_std / y_std) * u_y
        + (p_std / x_std) * p_x
        - (1 / Re) * ((u_std / (x_std**2)) * u_xx + (u_std / (y_std**2)) * u_yy)
    )
    equ_2 = (
        (v_std / t_std) * v_t
        + (u * u_std + u_mean) * (v_std / x_std) * v_x
        + (v * v_std + v_mean) * (v_std / y_std) * v_y
        + (p_std / y_std) * p_y
        - (1 / Re) * ((v_std / (x_std**2)) * v_xx + (v_std / (y_std**2)) * v_yy)
        - (input[:, 3] * ya0_std + ya0_mean)
        * L_adim
        * w_0**2
        * L_adim
        * torch.cos((w_0 * t_std * input[:, 2]) / (t_mean))
        / V_adim**2
    )
    equ_3 = (u_std / x_std) * u_x + (v_std / y_std) * v_y
    return equ_1, equ_2, equ_3


# Le NN


class MLP(nn.Module):
    def __init__(self, nb_entry, nb_neurons, nb_layers, nb_branches):
        super().__init__()
        self.init_layer = nn.ModuleList([nn.Linear(nb_entry, nb_neurons)])
        self.hiden_layers = nn.ModuleList(
            [nn.Linear(nb_neurons, nb_neurons) for _ in range(nb_layers - 1)]
        )
        self.final_layer = nn.ModuleList([nn.Linear(nb_neurons, nb_branches)])
        self.layers = self.init_layer + self.hiden_layers + self.final_layer
        self.initial_param()

    def forward(self, x):
        for k, layer in enumerate(self.layers):
            if k != len(self.layers) - 1:
                x = torch.tanh(layer(x))
            else:
                x = layer(x)
        return x  # Retourner la sortie

    def initial_param(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


class DeepONet(nn.Module):
    def __init__(
        self,
        nb_entry_branch,
        nb_entry_trunk,
        trunk_width,
        trunk_depth,
        branch_width,
        branch_depth,
        nb_branches,
        nb_exit,
    ):
        super().__init__()
        self.trunk = MLP(
            nb_entry_trunk, trunk_width, trunk_depth, nb_branches * nb_exit
        )
        self.branch = MLP(
            nb_entry_branch, branch_width, branch_depth, nb_branches * nb_exit
        )
        self.trunk.initial_param()
        self.branch.initial_param()
        self.nb_branches = nb_branches
        self.final_bias = nn.Parameter(torch.zeros(nb_exit))
        self.nb_exit = nb_exit

    def forward(self, x_trunk, x_branch):
        product_branch = self.trunk(x_trunk) * self.branch(x_branch)
        return (
            torch.sum(
                product_branch.reshape(self.nb_branches, -1, self.nb_exit),
                dim=0,
            )
            + self.final_bias
        )


if __name__ == "__main__":
    piche = DeepONet(
        nb_entry_branch=1,
        nb_entry_trunk=3,
        trunk_width=64,
        trunk_depth=6,
        branch_width=64,
        branch_depth=6,
        nb_branches=20,
        nb_exit=3,
    )
    nombre_parametres = sum(p.numel() for p in piche.parameters() if p.requires_grad)
    print(nombre_parametres)
