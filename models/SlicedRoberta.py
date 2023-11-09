import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers import RobertaModel

class SlicedRoberta(nn.Module):
    def __init__(self, num_labels: int = 2):
        super(SlicedRoberta, self).__init__()

        self.roberta = RobertaModel.from_pretrained("roberta-base", output_hidden_states=True)
        # freeze roberta model parameters
        for param in self.roberta.parameters():
            param.requires_grad = False

        self.num_layers = len(self.roberta.encoder.layer)
        self.hidden_dimension = self.roberta.encoder.layer[0].output.dense.weight.shape[0]
        self.num_labels = num_labels
        self.classification_layers = nn.ModuleList([nn.Linear(self.hidden_dimension, self.num_labels) for _ in range(self.num_layers)])
    
    def forward(self, input_ids, attention_mask: np.ndarray = None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        
        # The model outputs are returned in the form of a tuple. 
        # The first item in the tuple is the sequence of hidden-states at the output of the last layer.
        # The second item is the pooled output.
        # The third item is all the hidden states from all layers.
        all_hidden_states  = outputs[2] # shape: (num_layers, batch_size, sequence_length, hidden_dimension)

        # convert all_hidden_states from tuple to tensor
        all_hidden_states = torch.stack([tensor for tensor in all_hidden_states])

        ############# Use the first token of each sequence for classification #############

        # get the first position of tokens
        batch_size = all_hidden_states[0].shape[0]
        first_token_positions = np.zeros(shape=(batch_size,))
        
        # create a range array for the batch dimension
        batch_range = np.arange(all_hidden_states.shape[1])
        # select the first token (cls token) for every layer and batch
        new_hidden_states = all_hidden_states[:, batch_range, first_token_positions] # (num_layers, batch_size, hidden_dimension)

        # forward the hidden states to the classification layers
        # classification layer is of shape (hidden_dimension, num_labels)
        # new hidden_states is of shape (num_layers, batch_size, hidden_dimension)
        all_output_logits = []
        for hidden_state, classification_layer in zip(new_hidden_states, self.classification_layers):
            logits = classification_layer(hidden_state) # (batch_size, num_labels)
            all_output_logits.append(logits) # (num_layers, batch_size, num_labels)
        
        # convert to tensor
        all_output_logits_tensor = torch.stack(all_output_logits, dim=0)
        return all_output_logits_tensor # (num_layers, batch_size, num_labels)
    
def compute_loss(all_layer_logits: torch.Tensor, labels: torch.Tensor, num_layers: int, num_labels: int):
    """
    Arg:
        all_layer_logits <num_layers, batch_size, num_labels>: The predicted unnormalized logits of the model for all layers.
        labels <batch_size>: Ground truth class labels
    
    Returns:
        A tuple of summed_loss (1D tensor) and all_layer_loss (num_layers)
    """
    # replicate labels to all layers
    labels = labels.clone().detach().repeat(num_layers) # (batch_size) -> (num_layers, batch_size)

    ########### compute cross entropy loss ###########
    # collapse all dimensions for calculating loss
    labels = labels.reshape(-1) # (num_layers, batch_size) -> 1D tensor
    
    # collapse all dimensions except num_layers for calculating loss
    logits = all_layer_logits.reshape(-1, num_labels) # (num_layers, batch_size, num_labels) -> (num_labels)
    
    # calculate loss
    all_layer_loss = F.cross_entropy(logits, labels.long(), reduction='none') # (num_layers)
    
    # mean loss for backpropagation
    # e.g. loss = (u + v) / 2 -> loss with respective to the weights are u'/2 + v'/2
    summed_loss = all_layer_loss.mean() # (num_layers) -> (1D tensor)

    return summed_loss, all_layer_loss # (1D tensor), # (num_layers)