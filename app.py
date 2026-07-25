import streamlit as st

from utils import read_pdf

from rag import create_vector_store, ask_question

st.set_page_config(
    page_title="Advanced RAG",
    layout="wide"
)

st.title("📚 Advanced RAG Chatbot")

pdf = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if pdf:

    with st.spinner("Reading PDF..."):

        text = read_pdf(pdf)

        db = create_vector_store(text)

    st.success("Vector Database Created")

    question = st.chat_input("Ask Question")

    if question:

        answer, docs = ask_question(
            db,
            question
        )

        st.chat_message("user").write(question)

        st.chat_message("assistant").write(answer)

        with st.expander("Retrieved Chunks"):

            for i, doc in enumerate(docs):

                st.write(f"Chunk {i+1}")

                st.write(doc.page_content)
