import torch

class FeatureQueue():
    def __init__(self, dim, length):
        self.length = length
        self.queue = torch.zeros(length, dim)
        self.ptr = 0

    @torch.no_grad()
    def update(self, feat):

        batch_size = feat.shape[0]
        assert self.length % batch_size == 0  # for simplicity

        # replace the features at ptr (dequeue and enqueue)
        self.queue[self.ptr:self.ptr + batch_size] = feat
        self.ptr = (self.ptr + batch_size) % self.length  # move pointer

    def get(self):
        cnt = (self.queue[-1] != 0).sum()
        if cnt.item():
            return self.queue
        else:
            return None

class FeatureQueue_classwise():
    def __init__(self, dim, length):
        self.length = length
        self.queue = torch.zeros(length, dim)
        self.ptr = 0

    @torch.no_grad()
    def update(self, feat):

        batch_size = feat.shape[0]
        end_point = self.ptr+batch_size
        if end_point<self.length: # enough space
            # replace the features at ptr (dequeue and enqueue)
            self.queue[self.ptr:end_point] = feat
            
        else:
            self.queue[self.ptr:] = feat[:self.length-self.ptr]
            self.queue[:end_point % self.length] = feat[self.length-self.ptr:]

        self.ptr = (self.ptr + batch_size) % self.length  # move pointer


    def get(self):
        cnt = (self.queue[-1] != 0).sum()
        if cnt.item():
            return self.queue   #끝까지 꽉차있으면 return, 끝이 비어있으면 none
        else:
            return None

    def reset(self):
        self.queue = torch.zeros_like(self.queue)
