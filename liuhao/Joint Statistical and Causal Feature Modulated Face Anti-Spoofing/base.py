# device,
class BaseTrainer():
    def __init__(self, cfg, network, optimizer, criterion, lr_scheduler, trainloader, valloader, writer, pretrained):
        self.cfg = cfg
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        # self.device = device
        self.trainloader = trainloader
        self.valloader = valloader
        self.writer = writer
        self.pretrained = pretrained

    
    def load_model(self):
        raise NotImplementedError


    def save_model(self):
        raise NotImplementedError


    def train_one_epoch(self):
        raise NotImplementedError


    def train(self):
        raise NotImplementedError


    def validate(self):
        raise NotImplementedError