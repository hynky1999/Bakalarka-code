from torch.optim import AdamW
class TransformerAdamW(AdamW):
    no_decay = ["bias", "LayerNorm.weight"]

    def __init__(self, model, lr, weight_decay, eps, betas):
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in self.no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in self.no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        super().__init__(optimizer_grouped_parameters, lr=lr, eps=eps, betas=betas)
