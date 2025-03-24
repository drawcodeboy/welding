from .welding_dataset import WeldingDataset

def load_dataset(**cfg):
    if cfg['dataset'] == 'welding':
        return WeldingDataset(mode=cfg['mode'])