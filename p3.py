import os
import streamlit as st
import pdfplumber
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from markdownify import markdownify as md

# --- Load documents ---
@st.cache_data
def load_documents(folder_path="documents"):
    docs = []
    filenames = []
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                docs.append(f.read())
                filenames.append(file)
        elif file.endswith(".pdf"):
            text = ""
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            docs.append(text)
            filenames.append(file)
    return docs, filenames

docs, filenames = load_documents()

# --- Split documents into chunks for better retrieval ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_docs = []
for doc, fname in zip(docs, filenames):
    chunks = text_splitter.split_text(doc)
    all_docs.extend([Document(page_content=chunk, metadata={"source": fname}) for chunk in chunks])

# --- Create embeddings and build FAISS vector store ---
@st.cache_resource
def build_vectorstore(documents):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

vectorstore = build_vectorstore(all_docs)

# --- Streamlit UI ---
st.title("📚 Local Research Agent (LangChain)")

query = st.text_input("Enter your research query:")

if query:
    # Retrieve top-k relevant chunks
    results = vectorstore.similarity_search(query, k=3)
    
    combined_text = "\n".join([f"{res.metadata['source']}:\n{res.page_content}" for res in results])
    answer = f"### Query:\n{query}\n\n### Retrieved Context:\n{combined_text[:1500]}..."
    st.markdown(answer)
    
    st.subheader("Retrieved Documents")
    for res in results:
        st.write(f"- {res.metadata['source']}")
    
    if st.button("Export to Markdown"):
        summary = md(answer)
        with open("research_report.md", "w", encoding="utf-8") as f:
            f.write(summary)
        st.success("Exported as research_report.md")
