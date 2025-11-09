import os
import streamlit as st
import nltk
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


# -------------------- INITIAL SETUP --------------------
st.set_page_config(page_title="🧠 SmartBot: News Research Tool", page_icon="📰")

# Load environment variables
load_dotenv()

# Load OpenAI API key from environment or Streamlit Secrets
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not api_key or not api_key.startswith("sk-"):
    st.error(
        "❌ OpenAI API key not found. Set it in .env (for local use) "
        "or in Streamlit Secrets (for Cloud)."
    )
    st.stop()

# Make API key globally available to LangChain and OpenAI
os.environ["OPENAI_API_KEY"] = api_key

# Download required NLTK resources (quiet mode)
nltk.download("punkt_tab", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

# -------------------- STREAMLIT UI --------------------
st.title("🧠 SmartBot: News Research Tool")
st.sidebar.title("🔗 Enter News Article URLs")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")
index_folder = "faiss_index"
main_placeholder = st.empty()

# -------------------- PROCESS URLS (BUILD FAISS INDEX) --------------------
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

        embeddings = OpenAIEmbeddings()  # Uses API key from env
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
        if not os.path.exists(index_folder):
            st.warning("⚠️ Please process URLs first to create the FAISS index.")
            st.stop()

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            index_folder, embeddings, allow_dangerous_deserialization=True
        )

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
