import torch
from pprint import pprint
from omegaconf import OmegaConf
from IPython.display import Audio, display

torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
                               'latest_silero_models.yml',
                               progress=False)
models = OmegaConf.load('latest_silero_models.yml')
# see latest avaiable models
available_languages = list(models.tts_models.keys())
print(f'Available languages {available_languages}')

for lang in available_languages:
    _models = list(models.tts_models.get(lang).keys())
    print(f'Available models for {lang}: {_models}')