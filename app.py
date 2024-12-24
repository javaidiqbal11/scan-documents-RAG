import os
import json
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from utils.loader import load_documents
from utils.vector_store import create_vector_store
from utils.qa_chain import create_qa_chain

# Initialize FastAPI application
app = FastAPI()

# Globals for vector store and QA chain
vector_store = None
qa_chain = None

# Load OpenAI API key
def load_openai_api_key(filepath="api_key.json"):
    try:
        with open(filepath, "r") as file:
            data = json.load(file)
            return data.get("api_key")
    except Exception as e:
        raise RuntimeError(f"Error loading OpenAI API key: {e}")

# Initialize OpenAI API key
API_KEY_FILE = "api_key.json"
OPENAI_API_KEY = load_openai_api_key(API_KEY_FILE)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


@app.post("/upload-docs/")
async def upload_docs(files: list[UploadFile]):
    """Endpoint to upload and process Word documents."""
    global vector_store, qa_chain
    try:
        file_paths = []

        # Save uploaded files locally
        for file in files:
            file_path = f"./uploaded_{file.filename}"
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            file_paths.append(file_path)

        # Load and process documents
        documents = load_documents(file_paths)
        vector_store = create_vector_store(documents)
        qa_chain = create_qa_chain(vector_store)

        # Clean up uploaded files
        for file_path in file_paths:
            os.remove(file_path)

        return {"message": "Documents successfully loaded and processed!"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/query/")
async def query_docs(query: str = Form(...)):
    """Endpoint to query the processed documents."""
    if not qa_chain:
        return JSONResponse(
            status_code=400,
            content={"error": "Please upload and process documents first."},
        )
    try:
        response = qa_chain({"query": query})
        answer = response["result"]
        source_documents = response["source_documents"]

        # Extract and format the most relevant sources
        if source_documents:
            top_sources = "\n\n".join(
                [f"Source:\n{doc.page_content}" for doc in source_documents[:1]]
            )
        else:
            top_sources = "No relevant source documents found."

        return {"answer": answer, "related_source": top_sources}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

