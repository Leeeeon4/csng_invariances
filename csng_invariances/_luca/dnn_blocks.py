import torch.nn as nn


def act_func():
    act = nn.ModuleDict(
        [
            ["identity", nn.Identity()],
            ["relu", nn.ReLU()],
            ["lrelu", nn.LeakyReLU()],
            ["elu", nn.ELU()],
            ["selu", nn.SELU()],
            ["prelu", nn.PReLU()],
            ["softmax", nn.Softmax()],
            ["sigm", nn.Sigmoid()],
            ["tanh", nn.Tanh()],
        ]
    )
    return act


def FC_block(
    in_f,
    out_f,
    activation="relu",
    batchnorm=False,
    dropout_rate=0,
    order="abd",
    *args,
    **kwargs
):
    layers = [nn.Linear(in_f, out_f)]
    activations = act_func()
    keys_order = [char for char in order]
    layers_dict = {
        "a": activations[activation],
        "b": nn.BatchNorm1d(out_f),
        "d": nn.Dropout(dropout_rate),
    }
    if dropout_rate == 0:
        keys_order.remove("d")
    if batchnorm == False:
        keys_order.remove("b")
    for key in keys_order:
        layers.append(layers_dict[key])
    return nn.Sequential(*layers)

#TODO updade
def Conv1d_block(
    in_c,
    out_c,
    kernel_size,
    activation="relu",
    batchnorm=False,
    dropout_rate=0,
    *args,
    **kwargs
):
    layers = [nn.Conv1d(in_c, out_c, kernel_size, **kwargs)]
    activations = act_func()
    if batchnorm:
        layers.append(nn.BatchNorm1d(out_c))
    if dropout_rate != 0:
        layers.append(nn.Dropout(dropout_rate))
    layers.append(activations[activation])
    return nn.Sequential(*layers)


def Conv2d_block(
    in_c,
    out_c,
    kernel_size,
    activation="relu",
    batchnorm=False,
    dropout_rate=0,
    order="abd",
    *args,
    **kwargs
):
    layers = [nn.Conv2d(in_c, out_c, kernel_size, **kwargs)]
    activations = act_func()
    keys_order = [char for char in order]
    layers_dict = {
        "a": activations[activation],
        "b": nn.BatchNorm2d(out_c),
        "d": nn.Dropout(dropout_rate),
    }
    if dropout_rate == 0:
        keys_order.remove("d")
    if batchnorm == False:
        keys_order.remove("b")

    for key in keys_order:
        layers.append(layers_dict[key])
    return nn.Sequential(*layers)


def ConvTranspose2d_block(
    in_c,
    out_c,
    kernel_size,
    activation="relu",
    batchnorm=False,
    dropout_rate=0,
    order="abd",
    *args,
    **kwargs
):
    layers = [nn.ConvTranspose2d(in_c, out_c, kernel_size, **kwargs)]
    activations = act_func()
    keys_order = [char for char in order]
    layers_dict = {
        "a": activations[activation],
        "b": nn.BatchNorm2d(out_c),
        "d": nn.Dropout(dropout_rate),
    }
    if dropout_rate == 0:
        keys_order.remove("d")
    if batchnorm == False:
        keys_order.remove("b")

    for key in keys_order:
        layers.append(layers_dict[key])
    return nn.Sequential(*layers)
