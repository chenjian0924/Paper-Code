import torch
import torch.nn as nn

def Pruning(params, u_conv=1.1, pruning_rate=0.1,bias=0.1):
    modified_params = params.copy()
    for name, param in params.items():
        if 'conv' in name and 'weight' in name:
            weight = param.data.clone()
            out_channels = weight.shape[0]
            channel_sensitivity = []
            for idx in range(out_channels):
                w = weight[idx].reshape(weight.shape[1], -1)
                singular_values = torch.linalg.svdvals(w)
                max_singular = singular_values.max().item()
                channel_sensitivity.append(max_singular)
            channel_sensitivity = torch.tensor(channel_sensitivity)
            mean = channel_sensitivity.mean()
            std = channel_sensitivity.std()
            threshold = mean + u_conv * std
            index = torch.where(channel_sensitivity > threshold)[0]
            if index.numel() > 0:
                modified_params[name][index] = modified_params[name][index]*pruning_rate
                bias_name = name.replace('weight', 'bias')
                if bias_name in modified_params:
                    modified_params[bias_name][index] = bias
        elif ('fc' in name or 'classifier' in name) and 'weight' in name:
            weight = param.data.clone()
            out_features = weight.shape[0]
            neuron_sensitivity = []
            for idx in range(out_features):
                w = weight[idx]
                l2_norm = torch.norm(w, p=2).item()
                neuron_sensitivity.append(l2_norm)
            neuron_sensitivity = torch.tensor(neuron_sensitivity)
            mean = neuron_sensitivity.mean()
            std = neuron_sensitivity.std()
            threshold = mean + u_conv * std
            index = torch.where(neuron_sensitivity > threshold)[0]
            if index.numel() > 0:
                modified_params[name][index] = 0
                bias_name = name.replace('weight', 'bias')
                if bias_name in modified_params:
                    modified_params[bias_name][index] = 0
    return modified_params


