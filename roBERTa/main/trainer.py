from transformers import Trainer
from torch import nn, exp

class Fl_Trainer(Trainer):

    def __init__(self, alpha=0.25, gamma=2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

         # Учитываем только замаскированные токены (labels != -100)
        mask = (labels != -100)
        valid_logits = logits[mask]  # (num_masked_tokens, vocab_size)
        valid_labels = labels[mask]  # (num_masked_tokens)

        # Вычисляем Focal Loss только для валидных токенов
        ce_loss = nn.CrossEntropyLoss(reduction='none')(valid_logits, valid_labels)
        pt = exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt)**self.gamma * ce_loss).mean()

        return (focal_loss, outputs) if return_outputs else focal_loss