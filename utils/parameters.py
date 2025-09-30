import argparse


def get_parser():
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='The pytorch implementation for Visual Alignment Constraint '
                    'for Continuous Sign Language Recognition.')
    parser.add_argument(
        '--work-dir',
        # default='./work_dir/temp',
        default='./work_dir/phoenix2014_test/',
        help='the work folder for storing results')
    parser.add_argument(
        '--config',
        default='./configs/baseline.yaml',
        help='path to the configuration file')
    parser.add_argument(
        '--random_fix',
        type=str2bool,
        default=True,
        help='fix random seed or not')
    parser.add_argument(
        '--device',
        type=str,
        default=0,
        help='the indexes of GPUs for training or testing')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # processor
    parser.add_argument(
        '--phase', default='test', help='can be train, test and features')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # debug
    parser.add_argument(
        '--save-interval',
        type=int,
        default=200,
        help='the interval for storing models (#epochs)')
    parser.add_argument(
        '--random-seed',
        type=int,
        default=0,
        help='the default value for random seed.')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=100,
        help='the interval for evaluating models (#epochs)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=20,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--evaluate-tool', default="python", help='sclite or python')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # feeder
    parser.add_argument(
        '--feeder', default='dataloader_video.BaseFeeder', help='data loader will be used')
    parser.add_argument(
        '--dataset',
        # default=None,
        default='phoenix2014',
        help='data loader will be used'
    )
    parser.add_argument(
        '--dataset-info',
        default=dict(),
        help='data loader will be used'
    )
    parser.add_argument(
        '--num-worker',
        type=int,
        default=4,
        help='the number of worker for data loader')
    parser.add_argument(
        '--feeder-args',
        nargs='*',
        action=ParseKwargs,
        default=dict(),
        help='the arguments of data loader')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--slowfast-config', default=None, help='slowfast config')
    parser.add_argument(
        '--model-args',
        nargs='*',
        action=ParseKwargs,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--slowfast-args',
        nargs='*',
        action=ParseKwargs,
        default=dict(),
        help='the arguments of slowfast'
    )
    parser.add_argument(
        '--load-weights',
        # default=None,
        default='/media/cv3/store1/postgraduate/y2023/WLF/bestsr/best_checkpoints/phoenix2014_dev_18.01_test_18.28.pt',
        help='load weights for network initialization')
    parser.add_argument(
        '--load-checkpoints',
        default=None,
        help='load checkpoints for continue training')
    parser.add_argument(
        '--decode-mode',
        default="max",
        help='search mode for decode, max or beam')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # optim
    parser.add_argument(
        '--batch-size', type=int, default=16, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=8, help='test batch size')

    default_optimizer_dict = {
        "base_lr": 1e-2,
        "optimizer": "SGD",
        "nesterov": False,
        "step": [5, 10],
        "weight_decay": 0.00005,
        "start_epoch": 1,
    }
    default_loss_dict = {
        "SeqCTC": 1.0,
    }

    parser.add_argument(
        '--loss-weights',
        nargs='*',
        action=ParseKwargs,
        default=default_loss_dict,
        help='loss selection'
    )

    parser.add_argument(
        '--optimizer-args',
        nargs='*',
        action=ParseKwargs,
        default=default_optimizer_dict,
        help='the arguments of optimizer')

    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')

    # 新加
    # parser.add_argument('--train_dir', default='/media/cv3/store2/imagenet/', help='path to imagenet training set')
    parser.add_argument('--model_type', type=str, default='SLR', help='Model against GAN is trained: incv3, res50')
    parser.add_argument('--eps', type=int, default=0.03, help='Perturbation Budget')
    parser.add_argument('--target', type=int, default=-1, help='-1 if untargeted')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of trainig samples/batch')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2.25e-5, help='Initial learning rate for adam')
    parser.add_argument('--tk', type=float, default=0.6, help='path to checkpoint')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='path to checkpoint')
    return parser
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        for value in values:
            key, value = value.split('=')
            if value.isdecimal():
                value = int(value)
            else:
                try:
                    value = float(value)
                except:
                    pass
            getattr(namespace, self.dest)[key] = value

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
