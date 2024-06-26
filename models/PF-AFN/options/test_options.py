from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--warp_checkpoint', type=str, default='checkpoints/warp_viton.pth', help='load the pretrained model from the specified location')
        self.parser.add_argument('--gen_checkpoint', type=str, default='checkpoints/PFAFN/gen_model_final.pth', help='load the pretrained model from the specified location')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--unpaired', default=True, help='if enables, uses unpaired data from dataset')
        self.isTrain = False
