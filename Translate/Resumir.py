import easynlp
import gradio as gr


def execute(text):
    data = {
        "text": [
            text,
        ]
    }

    output_dataset = easynlp.summarization(data)

    return output_dataset["summarization"][0]


demo = gr.Interface(fn=execute,
                    inputs=[gr.Textbox(label="Ingrese un texto")],
                    outputs=[gr.Label(label="Resumen")]
                    )
demo.launch()
