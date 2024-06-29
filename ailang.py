import streamlit as s
import os
import constants
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS 
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI

# Function to create a session state
def get_session():
    if "session" not in s.session_state:
        s.session_state.session = {}
    return s.session_state.session

# Load session state
session_state = get_session()

s.set_page_config(page_title="Samast", page_icon=":world:")
s.header("SAMAST🤖")

os.environ["OPENAI_API_KEY"] = constants.API_KEY

with s.sidebar:
    s.subheader("Document Section📄")
    pdf_files = s.file_uploader("Upload Your Files Here: 📚🤓", accept_multiple_files=True)

query = s.text_input("Ask Your Question")  # question prompt

if query in session_state:
    s.write(session_state[query])
    s.stop()

c = s.button("🔎search")  # search button

if pdf_files and c:
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    vector_data = FAISS.from_texts(chunks, embeddings)

    docs = vector_data.similarity_search(query)

    index = VectorstoreIndexCreator().from_documents(docs)
    response = (index.query(query, llm=ChatOpenAI()))
    session_state[query] = response
    s.write(response)
