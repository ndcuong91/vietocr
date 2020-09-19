import matplotlib.pyplot as plt
from PIL import Image
import os, time

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import Levenshtein


def cer_loss_one_image(sim_pred, label):
    if (max(len(sim_pred), len(label)) > 0):
        loss = Levenshtein.distance(sim_pred, label) * 1.0 / max(len(sim_pred), len(label))
    else:
        return 0
    return loss


config = Cfg.load_config_from_name('vgg_transformer')

# config['weights'] = './weights/transformerocr.pth'
# config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['device'] = 'cuda:0'
config['predictor']['beamsearch'] = False
detector = Predictor(config)

from classifier_crnn.prepare_crnn_data import get_list_file_in_dir_and_subdirs

src_dir = '/home/duycuong/PycharmProjects/dataset/ocr/korea_test_set/korea_English_test/crnn_extend_False'

img_path = '/home/duycuong/PycharmProjects/dataset/ocr/train_data_29Feb_update_30Mar_13May_refined_23July/handwriting/' \
           'cleaned_data_02Mar/test/AICR_test1/AICR_P0000005/0005_1.jpg'
img_path = ''
if img_path == '':
    list_files = get_list_file_in_dir_and_subdirs(src_dir)
else:
    list_files = [img_path]

total_cer = 0
total_inference_time = 0
print('Total files:', len(list_files))
for idx, f in enumerate(list_files):
    img_path = os.path.join(src_dir, f)
    label_path = img_path.replace('.jpg', '.txt').replace('.png', '.txt').replace('.PNG', '.txt').replace('.JPG',
                                                                                                          '.txt')
    with open(label_path, 'r', encoding='utf-8') as f:
        label = f.readline()

    img = Image.open(img_path)
    begin = time.time()
    s = detector.predict(img)
    end = time.time()
    cer = cer_loss_one_image(label, s)
    total_cer += cer
    total_inference_time += (end - begin)
    print(idx, s, round(cer, 2), round(end - begin, 4))

print('avg cer: ', total_cer / len(list_files))
print('avg infer time: ', total_inference_time / len(list_files), ', fps:', len(list_files) /total_inference_time)

# # Download sample dataset

# In[9]:


# get_ipython().system(' gdown https://drive.google.com/uc?id=1W2PZC94sjpA1lS7FN33VoIVleSnnWOaA ')


# In[10]:


# get_ipython().system(' unzip -qq -o ./data.zip')


# # Train model

# 
# 
# 1.   Load your config
# 2.   Train model using your dataset above
# 
# 

# Load the default config, we adopt VGG for image feature extraction

# In[11]:
#
#
# from vietocr.tool.config import Cfg
# from vietocr.model.trainer import Trainer
#
#
# # # Change the config
# #
# # * *data_root*: the folder save your all images
# # * *train_annotation*: path to train annotation
# # * *valid_annotation*: path to valid annotation
# # * *print_every*: show train loss at every n steps
# # * *valid_every*: show validation loss at every n steps
# # * *iters*: number of iteration to train your model
# # * *export*: export weights to folder that you can use for inference
# # * *metrics*: number of sample in validation annotation you use for computing full_sequence_accuracy, for large dataset it will take too long, then you can reuduce this number
#
#
# config = Cfg.load_config_from_name('vgg_transformer')
# dataset_params = {
#     'name':'hw',
#     'data_root':'./data/',
#     'train_annotation':'train_annotation.txt',
#     'valid_annotation':'test_annotation.txt'
# }
#
# params = {
#          'print_every':200,
#          'valid_every':15*200,
#           'iters':20000,
#           'checkpoint':'./checkpoint/transformerocr_checkpoint.pth',
#           'export':'./weights/transformerocr.pth',
#           'metrics': 10000
#          }
#
# config['trainer'].update(params)
# config['dataset'].update(dataset_params)
# config['device'] = 'cuda:0'
#
#
# # you can change any of these params in this full list below
#
# # You should train model from our pretrained
# trainer = Trainer(config, pretrained=True)
#
#
# # Visualize your dataset to check data augmentation is appropriate
# trainer.visualize_dataset()
#
# # Train now
# trainer.train()
#
#
# # Visualize prediction from our trained model
# trainer.visualize_prediction()
#
#
# # Compute full seq accuracy for full valid dataset
# trainer.precision()
#
