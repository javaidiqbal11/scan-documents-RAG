from langchain.document_loaders import UnstructuredFileLoader

def load_documents(file_paths):
    """
    Load and preprocess documents from file paths.
    
    Args:
        file_paths (list of str): List of file paths to load.
    
    Returns:
        list: List of loaded documents.
    """
    documents = []
    for file_path in file_paths:
        loader = UnstructuredFileLoader(file_path)
        document = loader.load()
        documents.extend(document)
    return documents
