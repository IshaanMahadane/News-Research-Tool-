import os
import time
import pickle
import streamlit as st
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# -------------------- LOAD API KEY --------------------
load_dotenv()  # Load variables from .env
api_key = os.getenv("OPENAI_API_KEY")

if not api_key or not api_key.startswith("sk-"):
    st.error("❌ OpenAI API key not found. Please check your .env file.")
    st.stop()

# -------------------- STREAMLIT UI --------------------
st.title(" SmartBot: News Research Tool 📈")
st.sidebar.title("🔗 Enter News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url.strip())

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"
main_placeholder = st.empty()

# Initialize the OpenAI model
llm = OpenAI(temperature=0.9, max_tokens=500)

# -------------------- PROCESS URLS --------------------
if process_url_clicked:
    try:
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("📡 Loading data from URLs...")
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("✂️ Splitting text into chunks...")
        docs = text_splitter.split_documents(data)

        embeddings = OpenAIEmbeddings()
        main_placeholder.text("⚙️ Building FAISS index with OpenAI embeddings...")
        vectorstore_openai = FAISS.from_documents(docs, embeddings)

        # Save the FAISS index
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_openai, f)

        main_placeholder.success("✅ URLs processed successfully and FAISS index saved!")

    except Exception as e:
        st.error(f"❌ Error while processing URLs: {e}")

# -------------------- QUESTION ANSWERING --------------------
query = main_placeholder.text_input("💬 Ask a question about the processed articles:")

if query:
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)

            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )
            result = chain({"question": query}, return_only_outputs=True)

            st.header("🧾 Answer")
            st.write(result.get("answer", "No answer generated."))

            sources = result.get("sources", "")
            if sources:
                st.subheader("📚 Sources:")
                for source in sources.split("\n"):
                    st.write(source.strip())

        except Exception as e:
            st.error(f"❌ Error while retrieving answer: {e}")
    else:
        st.warning("⚠️ Please process the URLs first to build the FAISS index.")
