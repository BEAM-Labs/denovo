import torch_imputer
import torch
device = "cuda"
logits = [[[2, 10, 1, 1],[1,1,1,10],[1,1,7,1],[10,1,1,1]]]
l = torch.tensor(logits, dtype=float)
a = torch.nn.functional.log_softmax(l,dim= -1).to(device)
input_length = torch.tensor([4], dtype=int).to(device)
target = torch.tensor([[1,2,0]], dtype=int).to(device)
target_length = torch.tensor([3], dtype=int).to(device)
best_aligns_pad = torch_imputer.best_alignment(a.transpose(0,1), target, input_length, target_length, 3)
best_aligns_pad=torch.tensor(best_aligns_pad).to(device)
oracle_pos = (best_aligns_pad // 2).clip(max=target.shape[1] - 1).to(device)
oracle = target.gather(-1, oracle_pos)
oracle_empty = oracle.masked_fill(best_aligns_pad % 2 == 0, 3)
