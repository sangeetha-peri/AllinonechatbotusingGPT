import os
#works the best latest 
import asyncio
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from openai import AsyncOpenAI
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


# Load environment variables
load_dotenv()

# Initialize OpenAI Client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Streamlit App Configuration
st.set_page_config(page_title="Document Tools with GPT-4 Turbo", layout="wide")
st.title("Document Summarization, Chatbot, and Comparison using GPT-4 Turbo")
st.image("images/company_logo.png", width=200)

# Helper Functions
def clean_text(text):
    text = text.replace("\n", " ").strip()
    text = " ".join(text.split())  # Remove extra spaces
    return text

def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return clean_text(text)


async def recursive_summariser(client, text, model="gpt-4-turbo", input_chunk_tokens=5000, chunk_overlap=20, compression_ratio=5.0):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=input_chunk_tokens, chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_text(text)
    summaries = []
    for chunk in text_chunks:
        word_count = len(chunk.split(" "))
        desired_word_count = int(word_count // compression_ratio)
        query = f"Summarize the following text of {word_count} words to {desired_word_count} words:\n\n{chunk}"
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}]
        )
        summaries.append(response.choices[0].message.content)
    return " ".join(summaries)

async def chatbot_response(client, query, docs, model="gpt-4-turbo"):
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"""
    Using the provided context, answer the following question. If the answer is unavailable in the context, suggest potential search improvements for the user.

    Context:
    {context}
    Question:
    {query}
    Answer:
    """

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

async def compare_summaries(client, summary_1, summary_2, model="gpt-4-turbo"):
    query = f"""
    Compare the following summaries:
    Summary 1: {summary_1}
    Summary 2: {summary_2}
    """
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content

# Streamlit Application
tab1, tab2, tab3 = st.tabs(["Summarizer", "Chatbot", "Comparison"])

# Summarizer Tab
with tab1:
    st.header("Summarize PDF Documents")
    uploaded_files = st.file_uploader("Upload PDF files to summarize", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            st.write(f"Summarizing: {file.name}")
            summary = asyncio.run(recursive_summariser(openai_client, text))  # Use asyncio.run with the client instance
            st.subheader(f"Summary of {file.name}:")
            st.write(summary)

# Chatbot Tab
with tab2:
    st.header("Chat with Your PDFs")
    pdf_docs = st.file_uploader("Upload PDFs for chatbot", type=["pdf"], accept_multiple_files=True)
    if pdf_docs and st.button("Process PDFs"):
        raw_text = ""
        for pdf in pdf_docs:
            raw_text += extract_text_from_pdf(pdf)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        text_chunks = text_splitter.split_text(raw_text)

        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.success("PDFs processed successfully!")
    
    user_question = st.text_input("Ask a question about the uploaded PDFs:")
    if user_question:
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        new_db = FAISS.load_local("faiss_index", embeddings=embeddings)
        docs = new_db.similarity_search(user_question, k=5)
        combined_context = " ".join([doc.page_content for doc in docs])

        st.write("Retrieved Context:")
        st.write(combined_context)  # For debugging

        if combined_context.strip():
            response = asyncio.run(chatbot_response(openai_client, user_question, docs))
            st.write("Chatbot's reply:")
            st.write(response)
        else:
            st.write("No relevant context found. Please refine your question.")


# Comparison Tab
with tab3:
    st.header("Compare Tender Documents")
    folder_a_files = st.file_uploader("Upload Folder A (Business Tender Details)", type=["pdf"], accept_multiple_files=True)
    folder_b_files = st.file_uploader("Upload Folder B (Tender Quote Details)", type=["pdf"], accept_multiple_files=True)

    if folder_a_files and folder_b_files:
        folder_a_summaries = [asyncio.run(recursive_summariser(openai_client, extract_text_from_pdf(doc))) for doc in folder_a_files]
        folder_b_summaries = [asyncio.run(recursive_summariser(openai_client, extract_text_from_pdf(doc))) for doc in folder_b_files]

        for i, summary_b in enumerate(folder_b_summaries):
            for j, summary_a in enumerate(folder_a_summaries):
                comparison = asyncio.run(compare_summaries(openai_client, summary_a, summary_b))  # Use asyncio.run with the client instance
                st.write(f"Comparison between Document A{j+1} and Document B{i+1}:")
                st.write(comparison)
