import streamlit as st
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from utils import extract_text_from_pdf
from config import GROQ_API_KEY, GROQ_MODEL
import time

class PDFChatAssistant:
    def __init__(self, pdf_text):
        # Initialize once to avoid repeated embedding generation
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        
        # Preprocess text and create vector store
        self.pdf_text = pdf_text
        texts = self.text_splitter.split_text(pdf_text)
        self.vectorstore = FAISS.from_texts(texts, self.embeddings)
        
        # Initialize conversation history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

    def summarize_pdf(self) -> str:
        """Optimized summarization with error handling and timeout"""
        try:
            # Use a more specific prompt for better summaries
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert document summarizer. Provide a structured, concise summary highlighting key points, main arguments, and critical insights. Use clear, professional language."
                    },
                    {
                        "role": "user", 
                        "content": f"Summarize the following document, focusing on its core message and most important details. Limit to the most crucial information:\n\n{self.pdf_text[:4000]}"
                    }
                ],
                max_tokens=750,
                temperature=0.3  # More focused summary
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Summary generation error: {e}")
            return "Unable to generate summary. Please try again."

    def get_chat_response(self, user_query: str) -> str:
        """Enhanced chat response with context retrieval and conversation memory"""
        try:
            # Retrieve relevant context
            retrieved_docs = self.vectorstore.similarity_search(user_query, k=3)
            context = " ".join([doc.page_content for doc in retrieved_docs])

            # Prepare conversation history for context
            conversation_context = "\n".join([
                f"{'User' if i % 2 == 0 else 'Assistant'}: {msg}" 
                for i, msg in enumerate(st.session_state.chat_history[-4:])  # Last 2 exchanges
            ])

            # Generate response
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful PDF assistant. Provide conversational, precise answers based on the document context. If the query is not directly answerable from the context, guide the user helpfully."
                    },
                    {
                        "role": "user", 
                        "content": f"Previous Conversation:\n{conversation_context}\n\nDocument Context: {context}\n\nUser Query: {user_query}"
                    }
                ],
                max_tokens=500,
                temperature=0.5  # Balanced between creativity and precision
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Chat processing error: {e}")
            return "I'm having trouble processing your query. Could you rephrase?"

def main():
    st.title("📄 Intelligent PDF Assistant")
    st.sidebar.header("PDF Chat & Summarization")

    # PDF Upload
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])

    if uploaded_file is not None:
        # Extract PDF Text
        pdf_text = extract_text_from_pdf(uploaded_file)
        
        # Initialize PDF Assistant
        pdf_assistant = PDFChatAssistant(pdf_text)

        # Summarization Section
        st.sidebar.subheader("📋 Document Summary")
        if st.sidebar.button("Generate Summary"):
            with st.spinner('Generating summary...'):
                summary = pdf_assistant.summarize_pdf()
                st.write(summary)

        # Chat Section
        st.subheader("💬 Chat with PDF")
        
        # Display chat history
        for message in st.session_state.chat_history:
            st.chat_message("user" if st.session_state.chat_history.index(message) % 2 == 0 else "assistant").write(message)

        # User input
        if user_query := st.chat_input("Ask a question about the PDF"):
            # Display user message
            st.chat_message("user").write(user_query)
            st.session_state.chat_history.append(user_query)

            # Get and display response
            with st.spinner('Thinking...'):
                response = pdf_assistant.get_chat_response(user_query)
                st.chat_message("assistant").write(response)
                st.session_state.chat_history.append(response)

if __name__ == "__main__":
    main()