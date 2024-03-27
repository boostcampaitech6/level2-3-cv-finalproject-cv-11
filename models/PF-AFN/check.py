import torch
print(torch.cuda.is_available())  # True가 출력되어야 함
print(torch.__version__)
print(torch.version.cuda)