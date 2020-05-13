from transformerocr.tool.translate import build_model, translate, process_input, predict
from transformerocr.tool.utils import download_weights


class TextDetector():
    def __init__(self, config):
        download_weights(**config['weights'], quiet=True)

        model, vocab = build_model(config)

        self.config = config
        self.model = model
        self.vocab = vocab
        

    def predict(self, img):
        img = process_input(img)
        img = img.to(self.config['device'])

        s = translate(img, self.model)[0].tolist()
        s = self.vocab.decode(s)

        return s
