import torch

def init_tokenized(tokenizer):
    def collate_batch(batch):
        q_tensors, d1_tensors, d2_tensors = [], [], [] 
        for (q, d1, d2) in batch:
            q_in = tokenizer(q)
            q_in.input_ids += [103] * 8 # [MASK]
            q_in.attention_mask += [1] * 8
            q_in["input_ids"] = torch.LongTensor(q_in.input_ids).unsqueeze(0)
            q_in["attention_mask"] = torch.LongTensor(q_in.attention_mask).unsqueeze(0)
            q_tensors.append(q_in)
            d1_tensors.append(tokenizer(d1))
            d2_tensors.append(tokenizer(d2))
        
        return torch.cat(q_tensors), torch.cat(d1_tensors), torch.cat(d2_tensors)
    return collate_batch


