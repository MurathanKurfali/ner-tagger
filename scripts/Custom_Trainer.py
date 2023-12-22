import torch
from torch.nn import CrossEntropyLoss
from transformers import Trainer


# The custom Trainer class that inherits to override the loss function

class TrainerWeightedLoss(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
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
