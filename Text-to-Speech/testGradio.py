import gradio as gr
import matplotlib.pyplot as plt
import torch
import torchaudio
from PIL import Image


nombreArchivo = "output_wavernn.wav"
nombreImagen = "output_wavernn.jpeg"

tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
tacotron2 = tacotron2.to('cuda')
tacotron2.eval()


# Griffin-Lim
def execute(text):
    torch.random.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
    vocoder = bundle.get_vocoder().to(device)
    processor = bundle.get_text_processor()
    tacotron2 = bundle.get_tacotron2().to(device)

    with torch.inference_mode():
        processed, lengths = processor(text)
        processed = processed.to(device)
        lengths = lengths.to(device)
        spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
        waveforms, lengths = vocoder(spec, spec_lengths)
    torchaudio.save(nombreArchivo, waveforms[0:1].cpu(), sample_rate=vocoder.sample_rate)
    plt.imshow(spec[0].cpu().detach())
    plt.axis('off')
    plt.savefig("test.png", bbox_inches='tight')
    return [nombreArchivo, "test.png"]


demo = gr.Interface(fn=execute,
                    inputs=[gr.Textbox(label="Ingrese un texto")],
                    outputs=[gr.Audio(label="Output Box"), gr.Image(type="file", label="Espectro")]
                    )
demo.launch()
