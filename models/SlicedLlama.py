import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers import LlamaModel

class SlicedLlama(nn.Module):
    def __init__(self, num_labels: int = 2):
        super(SlicedLlama, self).__init__()

        self.llama = LlamaModel.from_pretrained("meta-llama/Llama-2-7b-hf", output_hidden_states=True)
        self.num_layers = len(self.llama.layers)
        self.hidden_dimension = self.llama.layers[0].mlp.gate_proj.weight.shape[1]
        self.num_labels = num_labels
        self.classification_layers = nn.ModuleList([nn.Linear(self.hidden_dimension, self.num_labels) for _ in range(self.num_layers)])
    
    def forward(self, input_ids, attention_mask: np.ndarray = None, token_type_ids: np.ndarray = None):
        outputs = self.llama(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # The model outputs are returned in the form of a tuple. 
        # The first item in the tuple is the sequence of hidden-states at the output of the last layer.
        # The second item is the pooled output.
        # The third item is all the hidden states from all layers.
        all_hidden_states  = outputs[2] # shape: (num_layers, batch_size, sequence_length, hidden_dimension)

        ############# Use the last token of each sequence for classification #############

        # get the last position of tokens
        if attention_mask is not None: # attention_mask shape: (batch_size, sequence_length)
            last_token_positions = attention_mask.argmin(-1) - 1 # (batch_size)
        else:
            non_padded_tokens = np.equal(input_ids, self.llama.config.pad_token_id) # (batch_size, sequence_length)
            last_token_positions = non_padded_tokens.argmin(-1) - 1 # (batch_size)

        # broadcast across num_layers dimension
        last_token_positions = last_token_positions[None, :] # (1, batch_size)
        
        # select the last token for every layer and batch
        new_hidden_states = all_hidden_states[:, :, last_token_positions, :] # (num_layers, batch_size, 1, hidden_dimension)

        # remove singleton dimension
        new_hidden_states = new_hidden_states.squeeze(2) # (num_layers, batch_size, hidden_dimension)

        # forward the hidden states to the classification layers
        # classification layer is of shape (hidden_dimension, num_labels)
        # new hidden_states is of shape (num_layers, batch_size, hidden_dimension)
        all_output_logits = []
        for hidden_state, classification_layer in zip(new_hidden_states, self.classification_layers):
            logits = classification_layer(hidden_state) # (batch_size, num_labels)
            all_output_logits.append(logits) # (num_layers, batch_size, num_labels)
        
        return all_output_logits # (num_layers, batch_size, num_labels)
    
def compute_loss(all_layer_logits: np.ndarray, labels: np.ndarray, num_layers: int, num_labels: int):
    """
    Arg:
        all_layer_logits <num_layers, batch_size, num_labels>: The predicted unnormalized logits of the model for all layers.
        labels <batch_size>: Ground truth class labels
    
    Returns:
        A tuple of summed_loss (1D tensor) and all_layer_loss (num_layers)
    """
    # replicate labels to all layers
    labels = torch.tensor(labels).repeat(num_layers) # (batch_size) -> (num_layers, batch_size)

    ########### compute cross entropy loss ###########
    # collapse all dimensions for calculating loss
    labels = labels.reshape(-1) # (num_layers, batch_size) -> 1D tensor
    
    # collapse all dimensions except num_layers for calculating loss
    logits = all_layer_logits.reshape(-1, num_labels) # (num_layers, batch_size, num_labels) -> (num_labels)
    
    # calculate loss
    all_layer_loss = F.cross_entropy(logits, labels, reduction='none') # (num_layers)
    
    # sum loss for backpropagation
    # sum is used so that the derivative of the loss are independent of each other
    # e.g. loss = u + v -> loss with respective to the weights are u' + v'
    summed_loss = all_layer_loss.sum() # (num_layers) -> (1D tensor)

    return summed_loss, all_layer_loss # (1D tensor), # (num_layers)