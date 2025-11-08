import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# -------------------- LOAD API KEY --------------------
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key or not api_key.startswith("sk-"):
    st.error("❌ OpenAI API key not found. Please check your .env file.")
    st.stop()

# -------------------- STREAMLIT UI --------------------
st.title("🧠 SmartBot: News Research Tool")
st.sidebar.title("🔗 Enter News Article URLs")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")
index_folder = "faiss_index"
main_placeholder = st.empty()

# -------------------- PROCESS URLS --------------------
if process_url_clicked:
    try:
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("📡 Loading data from URLs...")
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        main_placeholder.text("✂️ Splitting text into chunks...")
        docs = text_splitter.split_documents(data)

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        main_placeholder.text("⚙️ Building FAISS index with OpenAI embeddings...")
        vectorstore_openai = FAISS.from_documents(docs, embeddings)

        vectorstore_openai.save_local(index_folder)
        main_placeholder.success("✅ URLs processed and FAISS index saved!")

    except Exception as e:
        st.error(f"❌ Error while processing URLs: {e}")

# -------------------- QUESTION ANSWERING --------------------
query = main_placeholder.text_input("💬 Ask a question about the processed articles:")

if query:
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = FAISS.load_local(index_folder, embeddings, allow_dangerous_deserialization=True)

        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)

        prompt = ChatPromptTemplate.from_template(
            "Answer the question based on the context:\n\n{context}\n\nQuestion: {question}"
        )
        document_chain = prompt | llm | StrOutputParser()
        retriever_chain = RunnableParallel({
            "context": retriever,
            "question": RunnablePassthrough()
        })
        retrieval_chain = retriever_chain | document_chain

        result = retrieval_chain.invoke(query)

        st.header("🧾 Answer")
        st.write(result)

    except Exception as e:
        st.error(f"❌ Error while retrieving answer: {e}")
