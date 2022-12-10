from argparse import ArgumentParser

class EditImageOption:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--ckptpath', type=str, required=True, help='path to psp checkpoint')
        self.parser.add_argument('--encoder_type', type=str, default='GradualStyleEncoder')
        self.parser.add_argument('--input_nc', type=int, default=3)
        self.parser.add_argument('--output_size', type=int, default=1024)
        self.parser.add_argument('--start_from_latent_avg', type=bool, default=True)
        self.parser.add_argument('--learn_in_w', action='store_true')
        self.parser.add_argument('--device', type=str, default='cuda:0')

        self.parser.add_argument('--stylegan_size', default=1024, type=int,
                                help='size of pretrained StyleGAN Generator')

        self.parser.add_argument('--direction_dir', type=str, required=True)
        self.parser.add_argument('--attr_idx', type=int, default=20)
        self.parser.add_argument('--real', action='store_true')
        self.parser.add_argument('--image_path', type=str)
        self.parser.add_argument('--num_edit', type=int, default=100)
        self.parser.add_argument('--save_dir', type=str, required=True)

    def parse(self):
        opts = self.parser.parse_args()
        return opts