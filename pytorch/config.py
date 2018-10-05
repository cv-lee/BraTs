import torch

# Device Init
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model output shape Init
class_num = 2


# Data Handling Parameters
complete_threshold = 0.05
complete_rate = 0.66

core_threshold = 0.05
core_rate = 0.66

enhancing_threshold = 0.02
enhancing_rate = 0.7

# Data Augmentation Parameters
