from .DNN.dnn import DNN
from .LSTMNet.lstmnet import LSTMNet
from .AttentionNet.attentionnet import AttentionNet

def load_model(**cfg):
    if cfg['name'] == 'DNN':
        return DNN()
    elif cfg['name'] == 'LSTMNet':
        return LSTMNet()
    elif cfg['name'] == 'AttentionNet':
        return AttentionNet()