from dotenv import load_dotenv, find_dotenv
import os

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


load_dotenv(find_dotenv())
api_key = os.environ['OPENAI_API_KEY']

# Data Loader
pdf_path = "./pdfs/NIPS-2017-attention-is-all-you-need-Paper.pdf"
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

# Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
texts = text_splitter.split_documents(documents)

# Embeddings
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever()

# Generate
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Asking questions
print("Realiza una pregunta al documento")
while True:
    user_message = input("Pregunta: ")
    llm_response = qa(user_message)
    print("Respuesta: " + llm_response["result"])