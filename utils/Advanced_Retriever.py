from IPython.display import display, Markdown
from dotenv import load_dotenv, find_dotenv
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import chromadb
from langchain_community.vectorstores import Chroma
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.embeddings import SentenceTransformerEmbeddings


class AdvancedRetriever:
    def __init__(self):
        load_dotenv(find_dotenv())

    
    def parent_documents_retriever(self, loaded_docs):
        # This text splitter is used to create the parent documents
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        # This text splitter is used to create the child documents
        # It should create documents smaller than the parent
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        # The vectorstore to use to index the child chunks
        vectorstore = Chroma(
            collection_name="split_parents", embedding_function=SentenceTransformerEmbeddingFunction()
        )
        # The storage layer for the parent documents
        store = InMemoryStore()

        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

        retriever.add_documents(loaded_docs)

        return vectorstore, retriever
    
  

        

    

