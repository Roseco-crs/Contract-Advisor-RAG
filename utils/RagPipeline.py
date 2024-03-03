import streamlit as st
from dotenv import load_dotenv
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
from chain import Chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader



# Create an instance of MyRagFunctions
ragFunctions = MyRagFunctions()
parentDocumentRetriever = AdvancedRetriever()
chaiN = Chain()
  
def loading(path_file):
    return PyPDFLoader(path_file).load()


def handle_userinput(user_question):
    response = st.session_state.conversation_vec({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)



# Main Function
def main():
    load_dotenv()
    st.set_page_config(page_title="Powerfull Contract Advisor AI", page_icon= ":brain")

    st.write(css, unsafe_allow_html=True)

    if "conversation_vec" not in st.session_state:
        st.session_state.conversation_vec = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    with st.sidebar:
        pass
    
        st.markdown("## Upload Documents :law")
        external_data = st.file_uploader(" Upload your contract documents", accept_multiple_files= True)
        print(f"External datat{external_data}")
        if st.button("Retrieval"):
            with st.spinner("Processing the documents..."):
                # get contract documents
                # doc = ragFunctions.get_loaded_pdf(external_data)
                doc = ragFunctions.upload_and_load_langchain(external_data)                                  

                # get Parent Document Retriever
                vectorstore, retriever = parentDocumentRetriever.parent_documents_retriever(doc)

                # chain Rag Pipeline

                st.session_state.conversation_vec = chaiN.conversation_vec_chain(vectorstore)
                st.session_state.conversation_retr = chaiN.conversation_retr_chain(retriever)
                
                st.success("Documents processed successfully!")

    
 
    st.header("Your AI Contract Assistant")
    user_question = st.text_input("Ask a question to our AI Contract Advisor  ")
    if user_question:
        handle_userinput(user_question)
        
 

if __name__ == "__main__":
    main()