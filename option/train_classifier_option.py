from argparse import ArgumentParser

class TrainClassifierOpts:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--device', type=str, default='cuda:0')
        self.parser.add_argument('--lr', type=float, default=5e-5)
        self.parser.add_argument('--batch_size', type=int, default=8)
        self.parser.add_argument('--num_workers', type=int, default=2)
        self.parser.add_argument('--num_epochs', type=int, default=1)
        self.parser.add_argument('--cls_idx', type=int, default=0)
        self.parser.add_argument('--num_eval', type=int, default=2000)
        self.parser.add_argument('--num_train', type=int, default=float('inf'))

        self.parser.add_argument('--trainset_path', type=str, required=True)
        self.parser.add_argument('--testset_path', type=str, required=True)
        self.parser.add_argument('--image_list_path', type=str, required=True)
        self.parser.add_argument('--list_attr_celeba_path', type=str, required=True)
        self.parser.add_argument('--ckpt_dir', type=str, required=True)
        self.parser.add_argument('--log_dir', type=str, default='./log')

        self.parser.add_argument('--start', type=int, default=0)
        self.parser.add_argument('--end', type=int, default=40)

    def parse(self):
        opts = self.parser.parse_args()
        return opts

