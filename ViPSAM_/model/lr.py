import math
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, eta_min_ratio=0.0, last_epoch=-1):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return (step + 1) / num_warmup_steps
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        progress = min(1.0, progress)
        return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
    