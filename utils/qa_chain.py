from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def create_qa_chain(vector_store):
    """
    Create a RetrievalQA chain from a vector store.

    Args:
        vector_store (Chroma): Vector store instance.

    Returns:
        RetrievalQA: QA chain instance.
    """
    retriever = vector_store.as_retriever()

    # Define a custom prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}\nAnswer:",
    )

    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
    return qa_chain
