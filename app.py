import gradio as gr

from rag_exp import rag_chain


def chatbot(message):

    response = rag_chain(message)

    return response


demo = gr.Interface(
    fn = chatbot,
    inputs = gr.Textbox(
        lines = 2,
        placeholder = 'Ask a question'
    ),
    outputs = gr.Textbox(),
    title = 'ISOIL Exam Assistant'
)

demo.launch()