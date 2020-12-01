import matplotlib.pyplot as plt
from PIL import Image
import os, time, cv2
from classifier_crnn.prepare_crnn_data import get_list_file_in_dir_and_subdirs

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import Levenshtein


def cer_loss_one_image(sim_pred, label):
    if (max(len(sim_pred), len(label)) > 0):
        loss = Levenshtein.distance(sim_pred, label) * 1.0 / max(len(sim_pred), len(label))
    else:
        return 0
    return loss

debug = True
eval = True
config = Cfg.load_config_from_name('vgg_seq2seq') #vgg_transformer, vgg_seq2seq, vgg_convseq2seq

# config['weights'] = './weights/transformerocr.pth'
# config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['cnn']['pretrained'] = False
config['device'] = 'cuda:0'
config['predictor']['beamsearch'] = False
# config['dataset']['image_min_width'] = 64
# config['dataset']['image_max_width'] = 256
detector = Predictor(config)
src_dir = '/data20.04/data/data_Korea/Korea_test_Vietnamese_1106/crop'
#src_dir = '/data20.04/data/data_Korea/Cello_Vietnamese/crnn_extend_True_y_ratio_0.05_min_y_4_min_x_2'

img_path = '/home/duycuong/PycharmProjects/dataset/ocr/train_data_29Feb_update_30Mar_13May_refined_23July/handwriting/' \
           'cleaned_data_02Mar/test/AICR_test1/AICR_P0000005/0005_1.jpg'
img_path = ''
if img_path == '':
    list_files = get_list_file_in_dir_and_subdirs(src_dir)
    list_files = sorted(list_files)
else:
    list_files = [img_path]

total_cer = 0
total_inference_time = 0
print('Total files:', len(list_files))
for idx, f in enumerate(list_files):
    img_path = os.path.join(src_dir, f)
    label_path = img_path.replace('.jpg', '.txt').replace('.png', '.txt').replace('.PNG', '.txt').replace('.JPG',
                                                                                                          '.txt')
    if not os.path.exists(label_path):
        eval = False

    img = Image.open(img_path)
    begin = time.time()
    s = detector.predict(img)
    # print(s)
    # img.show()
    # time.sleep(15)
    end = time.time()
    if eval:
        with open(label_path, 'r', encoding='utf-8') as f:
            label = f.readline()
        cer = cer_loss_one_image(label, s)
        total_cer += cer
    total_inference_time += (end - begin)
    if debug:
        print(f,'.Predict:', s)
        img_cv = cv2.imread(img_path)
        cv2.imshow('img', img_cv)
        cv2.waitKey(0)
    else:
        if eval:
            print(idx, 'pred:', s, ', gt:', label, ', cer:', round(cer, 2), ', time:', round(end - begin, 4))

print('avg cer: ', total_cer / len(list_files))
print('avg infer time: ', total_inference_time / len(list_files), ', fps:', len(list_files) / total_inference_time)
