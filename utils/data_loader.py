import torch.utils.data


class DataLoader(torch.utils.data.Dataset):
    def __init__(self, code, channel, batch_size, dataset_size=1000, use_cuda=True):
        self.code, self.channel = code, channel
        self.batch_size, self.dataset_size = batch_size, dataset_size
        self.use_cuda = use_cuda

    def name(self):
        return ','.join([self.code.full_name(), self.channel.name])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        return self.next_batch()

    def next_batch(self, param_range=None, all_zero=True):
        res = self.channel.generate_sample(self.code, self.batch_size, param_range, all_zero)
        return tuple(v.cuda() for v in res) if self.use_cuda else res

    def generator(self, param_range=None, all_zero=True):
        for i in range(1, self.dataset_size + 1):
            yield i, self.next_batch(param_range, all_zero)
