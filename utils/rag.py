import streamlit as st
from dotenv import load_dotenv
from MyRagFunctions import MyRagFunctions
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain import OpenAI, VectorDBQA


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


# Main Function
def main():
    load_dotenv()
    st.set_page_config(page_title="Powerfull Contract Advisor AI", page_icon= ":brain")

    with st.sidebar:
        pass
    
    st.markdown("## Powerfull Contract Advisor AI :law")
    external_data = st.file_uploader(" Upload your contract documents", accept_multiple_files= True)
    if st.button("Retrieval"):
        with st.spinner("Processing"):
            # get contract documents
            #doc = get_text_from_miscrosoft_doc(external_data)
            doc = RagFunctions.get_text_from_pdf(external_data)
            #st.write(doc)

            # chunk the text
            chunk = RagFunctions.chunk_text(doc)
            #st.write(chunk[:3])

            # get embedding function or embedding
            openai_embedding = RagFunctions.get_OpenAIEmbeddings() 
            cache_embedding = get_cached_embedding()
            open_embedding = RagFunctions.get_open_embedding()         # Has an issue. 

            # get_faiss_vectorstore 
            #faiss_vectorstore_db = RagFunctions.get_faiss_vectorstore(chunk, cache_embedding)
    
            # get_chroma_vectorstore
            #chroma_vectorstore_db = RagFunctions.get_chroma_vectorstore(chunk, openai_embedding)
            chroma_vectorstore_db = RagFunctions.get_chroma_vectorstore(chunk, cache_embedding)
            #c = chroma_vectorstore_db.similarity_search("What is the best contract", k=4)
            #st.write(c)

        qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore = chroma_vectorstore_db)
        st.markdown("## Q&A")
        query = st.text_input("Type your question")
        answer = qa.run(query)
        st.write(answer)








if __name__ == "__main__":
    main()