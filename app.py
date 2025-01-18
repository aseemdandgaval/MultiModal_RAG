
import gradio as gr
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

from utils.app_utils import multi_modal_rag_chain

# Load the vector store and retriever
vectorstore = Chroma(collection_name="multi_modal_rag",
                     embedding_function=OpenAIEmbeddings(),
                     persist_directory="chroma_langchain_db")

id_key = "doc_id"
store = InMemoryStore()
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)
retriever = vectorstore.as_retriever()
chain_multimodal_rag = multi_modal_rag_chain(retriever)

def generate_response(message, history):
    """
    This function will be called for each new user message.
    We run the chain for the *latest user message only*.
    Then return the chain response as a string.
    """
    # Run the chain using the user message
    response_chunks = chain_multimodal_rag.invoke(message)

    # If the chain is streaming, it might return chunks.
    # We'll collect them into one final string for simplicity.
    if hasattr(response_chunks, "__iter__"):
        # It's a generator or list
        response_text = "".join(response_chunks)
    else:
        response_text = response_chunks

    # Return the final text
    return response_text

with gr.ChatInterface(
    fn=generate_response,
    title="Multi-modal RAG Chatbot",
    description="Ask a question about the LongNet paper.",
     examples=[
        {"text": "What is Dilated attention?"},
        {"text": "How is Dilated attention better than vanilla attention?"},
        {"text": "What is the difference between the computational cost of Dilated and Vanilla Attention?"}
     ],
) as demo:
    demo.launch()
