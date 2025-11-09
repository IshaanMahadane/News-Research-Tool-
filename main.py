import sys, types
if "tiktoken" not in sys.modules:
    sys.modules["tiktoken"] = types.SimpleNamespace(
        get_encoding=lambda name="gpt2": None,
        Encoding=lambda *a, **kw: None
    )

import os
import streamlit as st
import nltk
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

st.set_page_config(page_title="🧠 SmartBot: News Research Tool", page_icon="📰")
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not api_key or not api_key.startswith("sk-"):
    st.error("❌ OpenAI API key not found. Set it in .env or Streamlit Secrets.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key
nltk.download("punkt_tab", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

st.title("🧠 SmartBot: News Research Tool")
st.sidebar.title("🔗 Enter News Article URLs")
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")
index_folder = "chroma_db"
main_placeholder = st.empty()

if process_url_clicked:
    try:
        valid_urls = [u.strip() for u in urls if u.strip()]
        if not valid_urls:
            st.warning("⚠️ Please enter at least one valid URL.")
            st.stop()
        loader = UnstructuredURLLoader(urls=valid_urls)
        main_placeholder.text("📡 Loading data from URLs...")
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        main_placeholder.text("✂️ Splitting text into chunks...")
        docs = text_splitter.split_documents(data)
        embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
        main_placeholder.text("⚙️ Building Chroma index with OpenAI embeddings...")
        vectorstore_chroma = Chroma.from_documents(docs, embeddings, persist_directory=index_folder)
        vectorstore_chroma.persist()
        main_placeholder.success("✅ URLs processed and Chroma index saved!")
    except Exception as e:
        st.error(f"❌ Error while processing URLs: {e}")

query = main_placeholder.text_input("💬 Ask a question about the processed articles:")

if query:
    try:
        if not os.path.exists(index_folder):
            st.warning("⚠️ Please process URLs first to create the Chroma index.")
            st.stop()
        embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
        vectorstore = Chroma(persist_directory=index_folder, embedding_function=embeddings)
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        prompt = ChatPromptTemplate.from_template(
            "Answer the question based on the context:\n\n{context}\n\nQuestion: {question}"
        )
        document_chain = prompt | llm | StrOutputParser()
        retriever_chain = RunnableParallel({
            "context": retriever,
            "question": RunnablePassthrough(),
        })
        retrieval_chain = retriever_chain | document_chain
        result = retrieval_chain.invoke(query)
        st.header("🧾 Answer")
        st.write(result)
    except Exception as e:
        st.error(f"❌ Error while retrieving answer: {e}")
