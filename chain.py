from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory

import os

class Chain() :

    def __init__(self):
        os.environ["OPENAI_API_KEY"] = "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"

        books = os.listdir('data/markus_books_pdf')
        transcripts = os.listdir('data/markus_videos_pdf')

        book_loaders= [PyPDFLoader("data/markus_books_pdf/" + file_name) for file_name in books]
        transcript_loaders = [PyPDFLoader("data/markus_videos_pdf/" + file_name) for file_name in transcripts]
        loaders = book_loaders + transcript_loaders
        docs = []
        for loader in loaders:
            docs.extend(loader.load())


        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(documents, embeddings)

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self.chain = ConversationalRetrievalChain.from_llm(OpenAI(temperature=.04), vectorstore.as_retriever(), return_source_documents=True)