from PyPDF2 import PdfReader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Weaviate
from langchain_community.vectorstores import FAISS
# import os
# from dotenv import load_dotenv


# load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")



class MyRagFunctions:
    def __init__(self):
        pass
    

    def get_text_from_txtFile(self, files):
        """
        Get the text, based on the given .txt files

        :files: Files in which text will be extracted
        :return: The text from files
        """
        text = ""
        for file in files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    text += content
            except Exception as e:
                print(f"Error from reading file {file}: {e}")
        return text  
    

    def get_text_from_pdf(self, pdf):
        """
        Get text from the given pdf

        :pdf: pdf in which text will be extracted
        :return: The text from pdf
        """
        text = ""
        for doc in pdf:
            try: 
                reader = PdfReader(doc)
                for page in reader.pages:
                    text += page.extract_text()
            except Exception as e:
                print(f"Error from reading file {doc}: {e}")        
        return text
    

    def get_text_from_miscrosoft_doc(self, doc):
        """
        Get text from the given microsoft doc

        :doc: doc in which text will be extracted
        :return: The text from doc
        """
        try:
            loader = UnstructuredWordDocumentLoader(doc)
            text_data = loader.load()
        except Exception as e :
            print("Error from reading file {doc}: {e}")
            #text_data = "Error: Unable to extract text."    
        return text_data


    def chunk_text(self, text):
        """
        Chunk the given text
        :text: the text that will be chunk
        :return: chunks of text
        """
        try:
            text_siplitter = RecursiveCharacterTextSplitter(
                chunk_size = 100, 
                chunk_overlap = 50,
                separators=['\n', '\n\n'],
                length_function = len)
            chunk = text_siplitter.split_text(text)
        except Exception as e:
            print(f"Error coming from chunking function: {e}")
        return chunk
    

    def get_OpenAIEmbeddings(self):
        """
        Embedding
        :return: Openai embedding
        """
        return OpenAIEmbeddings()

    def get_HuggingFaceEmbeddings(self):
        """
        Embedding
        :return: HugginFaceEmbeddings
        """
        hf_embedding = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-xl", 
            model_kwargs={"device": "cpu"}
        )
        return hf_embedding
    

    def get_open_embedding(self):
        # create the open-source embedding function
        open_embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        return open_embedding
    

    def get_faiss_vectorstore(self, text_chunks, embedding_method):
        """
        Create a vector store or database and store embedding chunks into it.

        :text_chunks: chunks of the text
        :embedding_method: embedding method
        :return: vector database that contains embedded chunks of text
        """
        vectorstore = FAISS.from_texts(texts= text_chunks, embedding= embedding_method)
        #vectorstore = FAISS.from_documents(documents= text_chunks, embedding= embeddings_method)
        return vectorstore
    
    def get_chroma_vectorstore(self, text_chunks, embedding_method):
        """
        Create a vector store or database and store embedding chunks into it.
        
        :text_chunks: chunks of the text
        :embedding_method: embedding method
        :return: vector database that contains embedded chunks of text
        """
        # vectoestore = Chroma.from_documents(
        #     documents = text_chunks,
        #     embedding = embeddings_method,
        # )

        vectoestore = Chroma.from_texts(
            text_chunks, 
            embedding_method,
        )       
        return vectoestore
    
    def get_weaviate_vectorstore(self, text_chunks, embedding_method):
        vectorstore = Weaviate.from_texts(
            text_chunks,
            embedding_method,
            weaviate_url="http://127.0.0.1:8080"
        )


