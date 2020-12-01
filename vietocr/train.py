import argparse

from vietocr.model.trainer import Trainer
from vietocr.tool.config import Cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='see example at ')
    parser.add_argument('--checkpoint', required=False, help='your checkpoint')

    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)

    trainer = Trainer(config)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        
    trainer.train()


def train_customize():
    # 1.   Load your config
    # 2.   Train model using your dataset above

    # Load the default config, we adopt VGG for image feature extraction

    # * *data_root*: the folder save your all images
    # * *train_annotation*: path to train annotation
    # * *valid_annotation*: path to valid annotation
    # * *print_every*: show train loss at every n steps
    # * *valid_every*: show validation loss at every n steps
    # * *iters*: number of iteration to train your model
    # * *export*: export weights to folder that you can use for inference
    # * *metrics*: number of sample in validation annotation you use for computing full_sequence_accuracy, for large dataset it will take too long, then you can reuduce this number
    #

    config = Cfg.load_config_from_name('vgg_transformer')

    # config['vocab'] = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '

    dataset_params = {
        'name': 'hw',
        'data_root': './data_line/',
        'train_annotation': 'train_line_annotation.txt',
        'valid_annotation': 'test_line_annotation.txt'
    }

    params = {
        'print_every': 200,
        'valid_every': 15 * 200,
        'iters': 20000,
        'checkpoint': './checkpoint/transformerocr_checkpoint.pth',
        'export': './weights/transformerocr.pth',
        'metrics': 10000
    }

    config['trainer'].update(params)
    config['dataset'].update(dataset_params)
    config['device'] = 'cuda:0'

    # you can change any of these params in this full list below
    trainer = Trainer(config, pretrained=True)

    # Save model configuration for inference, load_config_from_file
    trainer.config.save('config.yml')

    # Visualize your dataset to check data augmentation is appropriate
    trainer.visualize_dataset()

    # Train now
    trainer.train()

    # Visualize prediction from our trained model
    trainer.visualize_prediction()

    # Compute full seq accuracy for full valid dataset
    trainer.precision()

if __name__ == '__main__':
    main()
