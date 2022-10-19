import gradio as gr
import os
import torch

device = torch.device('cpu')
torch.set_num_threads(4)
local_file = 'model.pt'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/es/v3_es.pt',
                                   local_file)

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)

example_text = 'Esto es un texto de prueba'
sample_rate = 48000
speaker = 'es_2'


def execute(text):
    audio_paths = model.save_wav(text=text,
                                 speaker=speaker,
                                 sample_rate=sample_rate)
    return audio_paths


demo = gr.Interface(fn=execute,
                    inputs=[gr.Textbox(label="Ingrese un texto")],
                    outputs=[gr.Audio(label="Output Box")]
                    )
demo.launch()
