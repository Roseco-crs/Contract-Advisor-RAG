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
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
import os
from dotenv import load_dotenv, find_dotenv
from io import BytesIO
from docx import Document
import io
import tempfile



class MyRagFunctions:
    def __init__(self):
        load_dotenv(find_dotenv())


    def upload_and_load_langchain(self, uploaded_files):
        #loaded_documents = []
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_extension = uploaded_file.name.split('.')[-1]

                if file_extension == 'txt':
                    # Load text file
                    text_content = uploaded_file.read().decode("utf-8")
                    loader = TextLoader(text_content)
                    loaded_document = loader.load()

                elif file_extension == 'pdf':
                    # Load PDF file using PyPDFLoader
                    # pdf_content = uploaded_file.read()
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        filename = uploaded_file.name
                        if filename.endswith('.pdf'):
                            loader = PyPDFLoader(tmp_file.name)

                            loaded_document = loader.load()
                            return loaded_document
                        
                    # loader = PyPDFLoader(pdf_content)
                    

                # elif file_extension == 'docx':
                #     # Load Word document using python-docx
                #     docx_content = uploaded_file.read()
                #     doc = Document(io.BytesIO(docx_content))
                #     text_content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                #     loader = TextLoader(text_content)
                #     loaded_document = loader.load()


                elif file_extension == 'zip':
                    # Load directory using DirectoryLoader
                    zip_content = uploaded_file.read()
                    loader = DirectoryLoader(zip_content)
                    loaded_document = loader.load()

                else:
                    print(f"Unsupported file type: {file_extension}")
                    continue

                # if loaded_document:
                #     loaded_documents.extend(loaded_document)
                

        return loaded_document
    

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
    
    def load_and_split_document(self, uploaded_file):
        """Loads and splits the document into pages."""
        if uploaded_file is not None:
            with BytesIO(uploaded_file.getbuffer()) as pdf_file:
                loader = PyPDFLoader(pdf_file)
                return loader.load_and_split()
        return None


    def get_loaded_pdf(self, pdf_file):
        """
        Get loaded pdf from the given pdf

        :pdf: pdf to load
        :return: loaded pdf
        """
        text = ""
        for doc in pdf_file:
            try: 
                reader = PdfReader(doc)
                for page in reader.pages:
                    text += page.extract_text()
            except Exception as e:
                print(f"Error from reading file {doc}: {e}")     
        return TextLoader(text).load()
        
        

        
    

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
    

    # def parent_child_chunk(self)
    

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

        vectorstore = Chroma.from_texts(
            text_chunks, 
            embedding_method,
        )       
        return vectorstore
    
    def get_weaviate_vectorstore(self, text_chunks, embedding_method):
        vectorstore = Weaviate.from_texts(
            text_chunks,
            embedding_method,
            weaviate_url="http://127.0.0.1:8080"
        )


