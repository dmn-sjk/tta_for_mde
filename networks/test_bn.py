from torch.nn import BatchNorm2d

def TestBN(BatchNorm2d):
    def __init__(self, **kwargs):
        super(TestBN, self).__init__(**kwargs)
        self.track_running_stats = False
        self.running_mean = None
        self.running_var = None

