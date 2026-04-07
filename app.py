import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# Load vector DB
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("faiss_index", embeddings)

# LLM
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0
)

# RAG Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=True
)

# UI
st.title("🤖 RAG Chatbot (LangChain)")

query = st.text_input("Ask something:")

if query:
    result = qa({"query": query})
    
    st.write("### 💬 Answer:")
    st.write(result["result"])

    st.write("### 📚 Sources:")
    for doc in result["source_documents"]:
        st.write(doc.metadata)
