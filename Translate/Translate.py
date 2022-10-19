import easynlp
import gradio as gr


def execute(text):
    data = {
        "text": [text]
    }
    output_language = "en"

    output_dataset = easynlp.translation(data, input_language="es", output_language=output_language)
    return output_dataset["translation"][0]


demo = gr.Interface(fn=execute,
                    inputs=[gr.Textbox(label="Ingrese un texto")],
                    outputs=[gr.Label(label="Traducido")]
                    )
demo.launch()
