import argparse
import sys
import os


class Opt(object):
    def __init__(self):
        super(Opt, self).__init__()

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--dir', type=str, help="tracking dataset")
        self.parser.add_argument('--output', type=str, default=None, help="dir to save the tracking result")

        self.parser.add_argument('--calib_path', type=str, help="calib dir")
        self.parser.add_argument('--is_plusai', type=bool, default=False, help="is PlusAI's dataset")
        self.parser.add_argument('--fov', type=int, default=120,
                                 help="fov of camera, values are [120, 150], default is 120")

    def parse_args(self, args=''):
        if args != '':
            opts = self.parser.parse_args(args)
        else:
            opts = self.parser.parse_args()
        print(opts.dir)
        if opts.dir is None or not os.path.isdir(opts.dir):
            self.parser.print_help()
            exit(0)

        return opts

    def init(self, args=''):
        opts = self.parse_args(args)
        return opts
