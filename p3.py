import os
import streamlit as st
import pdfplumber
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from markdownify import markdownify as md


# --- Load documents from uploaded files ---
@st.cache_data
def load_documents_from_upload(uploaded_files):
    docs = []
    filenames = []
    for uploaded_file in uploaded_files:
        fname = uploaded_file.name
        filenames.append(fname)
        if fname.endswith(".txt"):
            text = uploaded_file.getvalue().decode("utf-8")
            docs.append(text)
        elif fname.endswith(".pdf"):
            text = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            docs.append(text)
    return docs, filenames


st.title("📚 Local Research Agent (LangChain)")

st.sidebar.header("Upload Documents (PDF or TXT)")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more files", accept_multiple_files=True, type=["pdf", "txt"]
)


if uploaded_files:
    docs, filenames = load_documents_from_upload(uploaded_files)

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
else:
    st.info("Please upload at least one PDF or TXT file to get started.")
