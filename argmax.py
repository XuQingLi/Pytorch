import torch

outputs = torch.tensor([[0.2, 0.1], 
                        [0.3, 0.4]])
# argmax找到张量中某个维度上的最大值所在的索引
print("row_max:",outputs.argmax(0))
print("column_max:",outputs.argmax(1))
