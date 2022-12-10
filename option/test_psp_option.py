from argparse import ArgumentParser

class TestPspOpts:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--ckptpath', type=str, required=False, help='path to psp checkpoint')
        self.parser.add_argument('--encoder_type', type=str, default='GradualStyleEncoder')
        self.parser.add_argument('--input_nc', type=int, default=3)
        self.parser.add_argument('--output_size', type=int, default=1024)
        self.parser.add_argument('--start_from_latent_avg', type=bool, default=True)
        self.parser.add_argument('--learn_in_w', type=bool, default=False)
        self.parser.add_argument('--device', type=str, default='cuda:0')

    def parse(self):
        opts = self.parser.parse_args()
        return opts