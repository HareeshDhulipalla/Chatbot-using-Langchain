from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to user queries."),
    ("user", "Question: {question}")
])

# Use Groq-hosted LLaMA3 via OpenAI-compatible API
llm = ChatOpenAI(
    openai_api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
    model="llama3-70b-8192"
)

# Output parser
output_parser = StrOutputParser()

# Chain: prompt -> LLM -> parser
chain = prompt | llm | output_parser

# Streamlit UI
st.title("Langchain Chatbot with Groq (LLaMA3)")

input_text = st.text_input("Ask your question:")

if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)
