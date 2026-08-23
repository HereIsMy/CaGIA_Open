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


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.classifier(x)


def insert_transformer_adapters(model, reduction_factor, layer_start, layer_end):
    embedding_dim = model.config.hidden_size
    for i, layer in enumerate(model.encoder.layer[layer_start:layer_end]):
        layer.adapter = Adapter(embedding_dim, int(embedding_dim/reduction_factor))
        layer.forward_original = layer.forward

        def forward_with_adapter(self, *args, **kwargs):
            output = self.forward_original(*args, **kwargs)
            adapter_input = output[0]
            hidden_states = self.adapter(adapter_input)+adapter_input
            return (hidden_states,) + output[1:]
        layer.forward = types.MethodType(forward_with_adapter, layer)


def insert_classification_head(model, num_classes):
    embedding_dim = model.config.hidden_size
    model.classification_head = ClassificationHead(embedding_dim, num_classes)
    original_forward = model.forward

    def forward_with_classification_head(self, input_ids, *args, **kwargs):
        hidden_states = original_forward(input_ids, *args, **kwargs)
        hidden_states = hidden_states[0]
        pooled_output = hidden_states[:, 0, :]
        logits = self.classification_head(pooled_output)
        return logits
    model.forward = types.MethodType(forward_with_classification_head, model)


def freeze_model_parameters(model):
    for name, param in model.named_parameters():
        if 'adapter' in name or 'classification_head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return


def set_trainable_parameters(model):
    model.eval()
    for name, param in model.named_parameters():
        if 'adapter' in name or 'classification_head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    for name, module in model.named_modules():
        if 'adapter' in name or 'classification_head' in name:
            module.train()
