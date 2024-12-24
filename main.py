import os
import json
import gradio as gr
from utils.loader import load_documents
from utils.vector_store import create_vector_store
from utils.qa_chain import create_qa_chain

# Load OpenAI API key
def load_openai_api_key(filepath="api_key.json"):
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
            return data.get("api_key")
    except Exception as e:
        raise RuntimeError(f"Error loading OpenAI API key: {e}")

# Initialize OpenAI API key
API_KEY_FILE = "api_key.json"
OPENAI_API_KEY = load_openai_api_key(API_KEY_FILE)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Gradio App interface
def main():
    vector_store = None
    qa_chain = None

    def load_files(files):
        nonlocal vector_store, qa_chain
        try:
            file_paths = [file.name for file in files]
            documents = load_documents(file_paths)
            vector_store = create_vector_store(documents)
            qa_chain = create_qa_chain(vector_store)
            return "Documents successfully loaded and processed!"
        except Exception as e:
            return f"Error: {e}"

    def query_documents(query):
        if not qa_chain:
            return "Please upload and process documents first."
        try:
            response = qa_chain({"query": query})
            answer = response["result"]
            source_documents = response["source_documents"]

            # Extract and format the most relevant sources
            if source_documents:
                top_sources = "\n\n".join([f"Source:\n{doc.page_content}" for doc in source_documents[:1]])  # Show top 1 source
            else:
                top_sources = "No relevant source documents found."

            return f"Answer:\n{answer}\n\nRelated Source:\n{top_sources}"
        except Exception as e:
            return f"Error: {e}"

    with gr.Blocks() as app:
        gr.Markdown("# Document Query Interface\nUpload documents and query the content in a single screen.")

        with gr.Row():
            with gr.Column():
                file_uploader = gr.File(label="Upload Word Documents", file_types=[".docx"], file_count="multiple")
                load_button = gr.Button("Load Documents")
                load_status = gr.Textbox(label="Status", interactive=False)
            with gr.Column():
                query_input = gr.Textbox(label="Enter your query")
                query_button = gr.Button("Submit Query")
                query_output = gr.Textbox(label="Response", interactive=False)

        load_button.click(load_files, inputs=[file_uploader], outputs=[load_status])
        query_button.click(query_documents, inputs=[query_input], outputs=[query_output])

    app.launch(share=True)

if __name__ == "__main__":
    main()
