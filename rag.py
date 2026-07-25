from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate

from config import *

from prompt import PROMPT


embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

llm = ChatGroq(
    model=MODEL_NAME
)

prompt = PromptTemplate(
    input_variables=["context","question"],
    template=PROMPT
)


def create_vector_store(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_text(text)

    db = FAISS.from_texts(
        chunks,
        embeddings
    )

    return db


def ask_question(db, question):

    docs = db.similarity_search(
        question,
        k=TOP_K
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    chain = prompt | llm

    answer = chain.invoke(
        {
            "context":context,
            "question":question
        }
    )

    return answer.content, docs
