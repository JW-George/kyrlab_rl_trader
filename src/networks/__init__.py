import os

if os.environ.get('RLTRADER_BACKEND', 'pytorch') == 'pytorch':
    print('Enabling PyTorch...')
    from src.networks.networks_pytorch import Network, DNN, LSTMNetwork, CNN
else:
    print('Enabling TensorFlow...')
    from src.networks.networks_keras import Network, DNN, LSTMNetwork, CNN

__all__ = [
    'Network', 'DNN', 'LSTMNetwork', 'CNN'
]
