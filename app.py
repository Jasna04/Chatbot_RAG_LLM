from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma

# ==============================
# APP SETUP
# ==============================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==============================
# LLM
# ==============================

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.2,
    openai_api_key=OPENAI_API_KEY
)

# ==============================
# RAG KNOWLEDGE BASE
# ==============================

texts = [
    "This website provides web services and digital solutions.",
    "We help businesses build scalable products and platforms.",
    "Our services include product management, AI solutions, and analytics.",
    "You can contact us for consulting and support.",
    "We provide chatbot and AI-based automation solutions."
]

vectorstore = Chroma(
    persist_directory="db",
    embedding_function=OpenAIEmbeddings()
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# ==============================
# REQUEST MODEL
# ==============================

class ChatRequest(BaseModel):
    message: str

# ==============================
# ROUTES
# ==============================

@app.get("/")
def home():
    return {"message": "Chatbot API is running 🚀"}

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        response = qa_chain.run(req.message)
        return {"reply": response}
    except Exception as e:
        return {"reply": "Something went wrong"}
