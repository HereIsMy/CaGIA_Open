import torch.nn as nn
import types


class Adapter(nn.Module):
    def __init__(self, hidden_size, adapter_size):
        super(Adapter, self).__init__()
        self.adapter_down = nn.Linear(hidden_size, adapter_size)
        self.adapter_up = nn.Linear(adapter_size, hidden_size)
        self.act_fn = nn.ReLU()

    def forward(self, x):
        z = self.adapter_down(x)
        z = self.act_fn(z)
        z = self.adapter_up(z)
        return z


def insert_transformer_adapters(model, reduction_factor, layer_start, layer_end):
    embedding_dim = model.config.n_embd
    for i, layer in enumerate(model.transformer.h[layer_start:layer_end]):
        layer.adapter = Adapter(embedding_dim, int(embedding_dim/reduction_factor))
        layer.forward_original = layer.forward

        def forward_with_adapter(self, *args, **kwargs):
            output = self.forward_original(*args, **kwargs)
            adapter_input = output[0]
            hidden_states = self.adapter(adapter_input)+adapter_input
            return (hidden_states,) + output[1:]
        layer.forward = types.MethodType(forward_with_adapter, layer)


def set_trainable_parameters(model):
    model.eval()
    for name, param in model.named_parameters():
        if 'adapter' in name or 'score' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    for name, module in model.named_modules():
        if 'adapter' in name or 'score' in name:
            module.train()
