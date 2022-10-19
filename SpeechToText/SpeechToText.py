import gradio as gr
import torch
from glob import glob

device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU
model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='es',  # also available 'de', 'es'
                                       device=device)
(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils  # see function signature for details


# Griffin-Lim
def execute(text):
    print(text)
    test_files = glob(text)
    batches = split_into_batches(test_files, batch_size=10)
    input = prepare_model_input(read_batch(batches[0]),
                                device=device)

    output = model(input)
    respuesta = ""
    for example in output:
        respuesta += decoder(example.cpu())
    return respuesta


demo = gr.Interface(fn=execute,
                    inputs=[gr.Audio(label="Ingrese un audio", type="filepath", source="microphone", streaming="true")],
                    outputs=[gr.Textbox(label="Dijiste")]
                    , live="true2")
demo.launch()
