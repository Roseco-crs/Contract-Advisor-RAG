from MyRagFunctions import MyRagFunctions
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain import OpenAI, VectorDBQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from htmlTemplates import css, bot_template, user_template
from Advanced_Retriever import AdvancedRetriever
import os 
from dotenv import load_dotenv, find_dotenv




class Chain:
    def __init__(self):
        load_dotenv(find_dotenv())


    def conversation_vec_chain(self, vectorstore):
        """ 
        Args:
            vectorestore: vectorestore
        Return: conversation chain
        """
        llm = ChatOpenAI(temperature=0.0)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation_chain= ConversationalRetrievalChain.from_llm(
            llm = llm,
            memory = memory,
            retriever = vectorstore.as_retriever(),
            )
        return conversation_chain


    def conversation_retr_chain(self, retriever):
        """ 
        Args:
            retriever: retriever
        Return: conversation chain
        """

        llm = ChatOpenAI(temperature=0.0)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation_chain= ConversationalRetrievalChain.from_llm(
            llm = llm,
            memory = memory,
            retriever = retriever,
            )
        return conversation_chain   


    # def qNa_chain(self, vectorsotre):
    #     llm = ChatOpenAI(temperature=0.0)
    #     qa_stuff = RetrievalQA.from_chain_type(
    #         llm = llm,
    #         chain_type = "stuff",
    #         retriever = vectorsotre.as_retriever(),
    #         verbose = False,
    #     )
    #     return qa_stuff