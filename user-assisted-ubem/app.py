import streamlit as st

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate

# Title and Introduction
st.title("EP-Editor!")
st.write("Check this out!")
st.write("## Chat History")

# Initialize the model
model = ChatOllama(model='llama3.2:3b', base_url="http://localhost:11434/")

system_message_content = """Your task is to estimate the hourly occupancy schedule of a building by asking the building user a series of questions. The occupancy schedule should be represented on a scale from 0 (vacant) to 1 (fully occupied). Begin by asking a question. You may ask multiple questions in sequence if it helps refine your estimation. If the user is uncertain, provide an estimated percentage based on the available information. After each response and question, update your current estimation.

To guide you, here is an example format for the 24-hour occupancy schedule:
0, 0.1, 0.2, 0.1, 0.2, 0.1, 0, 0.1, 0.2, 0.1, 0.2, 0.1, 0, 0.1, 0.2, 0.1, 0.2, 0.1, 0, 0.1, 0.2, 0.1, 0.2, 0.1

Use this structure to update your estimations as you interact with the building user."""

system_message = SystemMessagePromptTemplate.from_template(system_message_content)

default_question = "Hello! What are the usual office hours? For example, is it 9 AM to 5 PM, or does it have extended or staggered hours?"

# Function to generate AI response
def generate_response(chat_history):
    chat_template = ChatPromptTemplate.from_messages(chat_history)
    chain = chat_template | model | StrOutputParser()
    response = chain.invoke({})
    return response


# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = [
        {"user": None, "assistant": default_question}  # Default AI message
    ]

# Function to clear chat history
def clear_history():
    st.session_state['chat_history'] = [
        {"user": None, "assistant": default_question}  # Reset to default message
    ]

# Function to get full chat history
def get_history():
    chat_history = [system_message]

    for chat in st.session_state['chat_history']:
        if chat['user']:  # Skip if the user message is None
            prompt = HumanMessagePromptTemplate.from_template(chat['user'])
            chat_history.append(prompt)

        if chat['assistant']:
            ai_message = AIMessagePromptTemplate.from_template(chat['assistant'])
            chat_history.append(ai_message)

    return chat_history

# Add a "Clear History" button
if st.button("Clear History and Restart"):
    clear_history()

# Streamlit form for user input
with st.form("llm-form"):
    text = st.text_area("Enter your text")
    submit = st.form_submit_button("Submit")

if submit and text:
    with st.spinner("Analyzing Your Building..."):
        # Add user input to history
        prompt = HumanMessagePromptTemplate.from_template(text)
        chat_history = get_history()
        chat_history.append(prompt)

        # Generate response from the model
        response = generate_response(chat_history)

        # Save the user input and response to session state
        st.session_state['chat_history'].append({'user': text, 'assistant': response})

# Display chat history
st.write('## Chat History')
for chat in reversed(st.session_state['chat_history']):
    if chat['user']:
        st.write(f"**👤 User**: {chat['user']}")
    if chat['assistant']:
        st.write(f"**🧠 EP-Editor**: {chat['assistant']}")
