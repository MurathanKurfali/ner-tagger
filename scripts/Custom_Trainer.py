import torch
from torch.nn import CrossEntropyLoss
from transformers import Trainer


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


# The custom Trainer class that inherits to override the loss function

class TrainerWeightedLoss(Trainer):

    def compute_loss2(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer.
        Overridden to implement weighted cross-entropy which may perform better on imbalanced datasets.
        """

        outputs = model(**inputs)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]
        logits = outputs["logits"]
        if labels is not None:
            # Define the loss function with optional class weights
            loss_fct = CrossEntropyLoss(weight=self.args.class_weights)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.args.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.args.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer.
        Overridden to implement Focal Loss, which may perform better on imbalanced datasets.
        """

        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]
        logits = outputs["logits"]

        if labels is not None:
            # Initialize Focal Loss function
            loss_fct = FocalLoss(weight=self.args.class_weights)

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.args.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.args.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss