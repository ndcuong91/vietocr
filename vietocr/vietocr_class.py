from PIL import Image
import os, time, cv2
from classifier_crnn.prepare_crnn_data import get_list_file_in_dir_and_subdirs

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

debug = False
eval = True

# total_cer = 0
# total_inference_time = 0
# print('Total files:', len(list_files))
# for idx, f in enumerate(list_files):
#     img_path = os.path.join(src_dir, f)
#     label_path = img_path.replace('.jpg', '.txt').replace('.png', '.txt').replace('.PNG', '.txt').replace('.JPG',
#                                                                                                           '.txt')
#     if not os.path.exists(label_path):
#         eval = False
#
#     img = Image.open(img_path)
#     Image.fromarray()
#     begin = time.time()
#     s = detector.predict(img)
#     end = time.time()
#     if eval:
#         with open(label_path, 'r', encoding='utf-8') as f:
#             label = f.readline()
#         cer = cer_loss_one_image(label, s)
#         total_cer += cer
#     total_inference_time += (end - begin)
#     if debug:
#         print('Predict:', s)
#         img_cv = cv2.imread(img_path)
#         cv2.imshow('img', img_cv)
#         cv2.waitKey(0)
#     else:
#         if eval:
#             print(idx, 'pred:', s, ', gt:', label, ', cer:', round(cer, 2), ', time:', round(end - begin, 4))
#
# print('avg cer: ', total_cer / len(list_files))
# print('avg infer time: ', total_inference_time / len(list_files), ', fps:', len(list_files) / total_inference_time)


class Classifier_Vietocr:
    def __init__(self, ckpt_path=None, gpu='0',
                 config_name='vgg_seq2seq', write_file=False, debug=False):
        print('Classifier_Vietocr. Init')
        self.config = Cfg.load_config_from_name(config_name)

        # config['weights'] = './weights/transformerocr.pth'
        if ckpt_path is not None:
            self.config['weights'] = ckpt_path
        self.config['cnn']['pretrained'] = False
        if gpu is not None:
            self.config['device'] = 'cuda:' + str(gpu)
        self.config['predictor']['beamsearch'] = False
        self.model = Predictor(self.config)

    def inference(self, numpy_list, debug=False):
        print('Classifier_Vietocr. Inference',len(numpy_list),'boxes')
        text_values = []
        prob_value = []
        # t = tqdm(iter(val_loader), total=len(val_loader), desc='Classifier_CRNN. Inference...')
        for idx, f in enumerate(numpy_list):
            if idx <70:
                continue
            img = Image.fromarray(f)
            #img.show()
            #time.sleep(4)
            s = self.model.predict(img)
            if debug:
                print(idx, s)
                cv2.imshow('sample',f)
                cv2.waitKey(0)
            text_values.append(s)
        return text_values, prob_value


def test_inference():
    engine = Classifier_Vietocr(gpu='0',
                                write_file=False,
                                debug=False)

    begin = time.time()
    src_dir = '/data20.04/data/data_Korea/Korea_test_Vietnamese_1106/vietnam1'
    src_dir = '/data20.04/data/aicr/train_data_29Feb_update_30Mar_13May_refined_13Nov/handwriting/cleaned_data_02Mar/test'

    img_path = '/home/duycuong/PycharmProjects/dataset/ocr/train_data_29Feb_update_30Mar_13May_refined_23July/handwriting/' \
               'cleaned_data_02Mar/test/AICR_test1/AICR_P0000005/0005_1.jpg'
    img_path = ''
    if img_path == '':
        list_files = get_list_file_in_dir_and_subdirs(src_dir)
        list_files = [os.path.join(src_dir,f) for f in list_files]
    else:
        list_files = [img_path]

    numpy_list=[]
    for file in list_files:
        print(file)
        cv_img = cv2.imread(file)
        numpy_list.append(cv_img)
    a, b = engine.inference(numpy_list, debug=True)
    end = time.time()
    print('Inference time:', end - begin, 'seconds')


if __name__ == "__main__":
    # ample_codes()
    test_inference()
