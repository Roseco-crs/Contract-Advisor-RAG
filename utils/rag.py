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


# Create an instance of MyRagFunctions
RagFunctions = MyRagFunctions()

store = LocalFileStore("./embeddings_cache")
# store = InMemoryByteStore()


def get_cached_embedding():
    # create an embedder
    embedding = RagFunctions.get_OpenAIEmbeddings()
    cache_embedding = CacheBackedEmbeddings.from_bytes_store(
        embedding,
        store,
        namespace = embedding.model
           )
    return cache_embedding


def conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain= ConversationalRetrievalChain.from_llm(
        llm = llm,
        memory = memory,
        retriever = vectorstore.as_retriever(),
        )
    return conversation_chain

def qNa_chain(vectorsotre):
    llm = ChatOpenAI(temperature=0.0)
    qa_stuff = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = vectorsotre.as_retriever(),
        verbose = False,
    )
    return qa_stuff
     



# Main Function
def main():
    load_dotenv()
    st.set_page_config(page_title="Powerfull Contract Advisor AI", page_icon= ":brain")

    with st.sidebar:
        pass
    
    st.markdown("## Powerfull Contract Advisor AI :law")
    external_data = st.file_uploader(" Upload your contract documents", accept_multiple_files= True)
    if st.button("Retrieval"):
        with st.spinner("Processing the documents..."):
            # get contract documents
            doc = RagFunctions.get_text_from_pdf(external_data)

            # chunk the text
            chunk = RagFunctions.chunk_text(doc)

            # get embedding function or embedding
            openai_embedding = RagFunctions.get_OpenAIEmbeddings() 
            cache_embedding = get_cached_embedding()
            open_embedding = RagFunctions.get_open_embedding()         # Has an issue. 

            # get_faiss_vectorstore 
            #faiss_vectorstore_db = RagFunctions.get_faiss_vectorstore(chunk, cache_embedding)
    
            # get_chroma_vectorstore
            chroma_vectorstore_db = RagFunctions.get_chroma_vectorstore(chunk, openai_embedding)
            #chroma_vectorstore_db = RagFunctions.get_chroma_vectorstore(chunk, cache_embedding) 

            conversation= conversation_chain(chroma_vectorstore_db)
            qa_chain = qNa_chain(chroma_vectorstore_db)
            st.success("Documents processed successfully!")
        
    st.markdown("## Q&A")
    question = st.text_input("Type your question")
    if question and external_data:
        with st.spinner("Generating answer..."):
            try:
                result = qa_chain.invoke(question)
                st.write(" Answer: ", result["result"])
            except Exception as e:
                st.error(f"There is an error in the processing of the question: {e}")


if __name__ == "__main__":
    main()