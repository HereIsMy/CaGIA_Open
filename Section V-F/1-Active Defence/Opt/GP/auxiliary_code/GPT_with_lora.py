import torch.nn as nn
import types


class LoRA(nn.Module):
    def __init__(self, in_features, adapter_size, out_features):
        super(LoRA, self).__init__()
        self.adapter_down = nn.Linear(in_features, adapter_size)
        self.adapter_up = nn.Linear(adapter_size, out_features)

    def forward(self, x):
        z = self.adapter_down(x)
        z = self.adapter_up(z)
        return z


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.classifier(x)


def insert_c_attn_lora(model, reduction_factor):

    embedding_dim = model.config.n_embd

    for i, layer in enumerate(model.h):
        c_attn = layer.attn.c_attn
        c_attn.lora_layer = LoRA(embedding_dim, int(embedding_dim/reduction_factor), embedding_dim*3)
        c_attn.forward_original = c_attn.forward

        def forward_with_c_attn_lora(self, hidden_states):
            output = self.forward_original(hidden_states)
            lora_output = self.lora_layer(hidden_states)
            hidden_states = output + lora_output
            return hidden_states

        c_attn.forward = types.MethodType(forward_with_c_attn_lora, c_attn)


def insert_ac_proj_lora(model, reduction_factor):

    embedding_dim = model.config.n_embd

    for i, layer in enumerate(model.h):
        ac_proj = layer.attn.c_proj
        ac_proj.lora_layer = LoRA(embedding_dim, int(embedding_dim/reduction_factor), embedding_dim)
        ac_proj.forward_original = ac_proj.forward

        def forward_with_ac_proj_lora(self, hidden_states):
            output = self.forward_original(hidden_states)
            lora_output = self.lora_layer(hidden_states)
            hidden_states = output + lora_output
            return hidden_states

        ac_proj.forward = types.MethodType(forward_with_ac_proj_lora, ac_proj)


def insert_c_fc_lora(model, reduction_factor):

    embedding_dim = model.config.n_embd

    for i, layer in enumerate(model.h):
        c_fc = layer.mlp.c_fc
        c_fc.lora_layer = LoRA(embedding_dim, int(embedding_dim/reduction_factor), embedding_dim*4)
        c_fc.forward_original = c_fc.forward

        def forward_with_c_fc_lora(self, hidden_states):
            output = self.forward_original(hidden_states)
            lora_output = self.lora_layer(hidden_states)
            hidden_states = output + lora_output
            return hidden_states

        c_fc.forward = types.MethodType(forward_with_c_fc_lora, c_fc)


def insert_mc_proj_lora(model, reduction_factor):

    embedding_dim = model.config.n_embd

    for i, layer in enumerate(model.h):
        mc_proj = layer.mlp.c_proj
        mc_proj.lora_layer = LoRA(embedding_dim*4, int(embedding_dim*4/reduction_factor), embedding_dim)
        mc_proj.forward_original = mc_proj.forward

        def forward_with_mc_proj_lora(self, hidden_states):
            output = self.forward_original(hidden_states)
            lora_output = self.lora_layer(hidden_states)
            hidden_states = output + lora_output
            return hidden_states

        mc_proj.forward = types.MethodType(forward_with_mc_proj_lora, mc_proj)


def insert_classification_head(model, num_classes):
    embedding_dim = model.config.n_embd
    model.classification_head = ClassificationHead(embedding_dim, num_classes)
    original_forward = model.forward

    def forward_with_classification_head(self, input_ids, *args, **kwargs):
        hidden_states = original_forward(input_ids, *args, **kwargs)
        hidden_states = hidden_states[0]
        pooled_output = hidden_states[:, -1, :]
        logits = self.classification_head(pooled_output)
        return logits
    model.forward = types.MethodType(forward_with_classification_head, model)


def freeze_model_lora_parameters(model):
    for name, param in model.named_parameters():
        if 'lora_layer' in name or 'classification_head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
