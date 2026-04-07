import os
import csv
import re
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# LangChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


# =====================================================
# APP SETUP
# =====================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# REQUEST MODEL
# =====================================================

class ChatInput(BaseModel):
    message: str
    site: Optional[str] = "default"


BASE_DIR = os.path.dirname(__file__)
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")


# =====================================================
# SENDGRID
# =====================================================

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
FROM_EMAIL = os.getenv("FROM_EMAIL")
SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL")


def send_support_email(subject: str, body: str):
    if not SENDGRID_API_KEY:
        return

    message = Mail(
        from_email=FROM_EMAIL,
        to_emails=SUPPORT_EMAIL,
        subject=subject,
        plain_text_content=body,
    )

    try:
        SendGridAPIClient(SENDGRID_API_KEY).send(message)
    except Exception as e:
        print("SendGrid error:", e)


def create_support_ticket(site: str, user_message: str):
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"

    body = f"""
Ticket ID: {ticket_id}
Store: {site.upper()}
Time: {datetime.utcnow().isoformat()}

Message:
{user_message}
"""

    send_support_email(f"[{site}] Support Ticket {ticket_id}", body)
    return ticket_id


def detect_escalation(msg: str):
    return any(k in msg for k in [
        "human", "agent", "support", "complaint",
        "issue", "problem", "help", "escalate"
    ])


# =====================================================
# DATA LOADING
# =====================================================

def load_csv(filename):
    path = os.path.join(KNOWLEDGE_DIR, filename)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


CHRISTMAS_ORDERS = load_csv("orders.csv")
PARIS_ORDERS = load_csv("paris_orders.csv")
PARIS_PRODUCTS = load_csv("womens_collections.csv")


# =====================================================
# LANGCHAIN SETUP (NEW)
# =====================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.2,
    openai_api_key=OPENAI_API_KEY
)

# Build knowledge base dynamically
texts = (
    [str(o) for o in CHRISTMAS_ORDERS] +
    [str(o) for o in PARIS_ORDERS] +
    [str(p) for p in PARIS_PRODUCTS]
)

vectorstore = Chroma.from_texts(texts, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


# =====================================================
# HELPERS
# =====================================================

def find_order(message, orders):
    for word in re.findall(r"[A-Za-z0-9\-]+", message.lower()):
        for o in orders:
            if o["OrderID"].lower() == word:
                return o
    return None


# =====================================================
# CHAT ENDPOINT
# =====================================================

@app.post("/chat")
def chat(data: ChatInput):
    msg = data.message.lower()
    site = data.site.lower()

    # 🔴 ESCALATION
    if detect_escalation(msg):
        ticket = create_support_ticket(site, data.message)
        return {"reply": f"Ticket created: {ticket}"}

    # 👋 GREETING
    if msg in ["hi", "hello", "hey"]:
        return {"reply": f"Hi! Welcome to {site} store."}

    # =================================================
    # ORDERS
    # =================================================

    orders = CHRISTMAS_ORDERS if site != "paris" else PARIS_ORDERS
    order = find_order(msg, orders)

    if order:
        return {
            "reply": f"""
Order {order['OrderID']}
Item: {order.get('ItemName')}
Status: {order.get('OrderStatus')}
Amount: {order.get('TotalAmount', order.get('TotalAmountEUR'))}
"""
        }

    # =================================================
    # PRODUCTS (PARIS)
    # =================================================

    if site == "paris":
        for p in PARIS_PRODUCTS:
            if p["product_name"].lower() in msg:
                return {
                    "reply": f"{p['product_name']} costs €{p['price_eur']}"
                }

    # =================================================
    # 🤖 LANGCHAIN FALLBACK (CORE UPGRADE)
    # =================================================

    try:
        ai_response = qa_chain.run(data.message)
        return {"reply": ai_response}
    except Exception as e:
        print("AI error:", e)

    return {"reply": "I can help with orders, products, or support."}
