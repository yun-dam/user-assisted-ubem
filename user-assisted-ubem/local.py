from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

# Initialize the model
model = ChatOllama(model='llama3.2:3b', base_url="http://localhost:11434/")

# System message content
system_message_content = """Your task is to estimate the hourly occupancy schedule of a building by asking the building user a series of questions. The occupancy schedule should be represented on a scale from 0 (vacant) to 1 (fully occupied). Begin by asking a question. You may ask multiple questions in sequence if it helps refine your estimation. If the user is uncertain, provide an estimated percentage based on the available information. After each response and question, update your current estimation.

To guide you, here is an example format for the 24-hour occupancy schedule:
[0, 0.1, 0.2, 0.1, 0.2, 0.1, 0, 0.1, 0.2, 0.1, 0.2, 0.1, 0, 0.1, 0.2, 0.1, 0.2, 0.1, 0, 0.1, 0.2, 0.1, 0.2, 0.1]

Use this structure to update your estimations as you interact with the building user."""

system_message = SystemMessagePromptTemplate.from_template(system_message_content)

# Chat history initialization
chat_history = []

# Function to generate AI response
def generate_response(input_text):
    """
    Sends the input text to the AI model and returns the response.
    """
    response = model.invoke(input_text)
    return response.content

# Function to prepare the full chat history
def get_history():
    """
    Combines the system message and the chat history into a single structured list.
    """
    history = [system_message]

    for chat in chat_history:
        if chat['user']:  # Add user messages
            prompt = HumanMessagePromptTemplate.from_template(chat['user'])
            history.append(prompt)

        if chat['assistant']:  # Add assistant responses
            ai_message = AIMessagePromptTemplate.from_template(chat['assistant'])
            history.append(ai_message)

    return history

# Function to initialize chat with the AI
def initialize_chat():
    """
    Initializes the chat by generating the first response to the system message.
    """
    # Generate the initial response from the AI
    input_text = system_message_content  # Start with the system message content
    initial_response = generate_response(input_text)

    # Add the system message and AI's response to chat history
    chat_history.append({"user": None, "assistant": initial_response})
    print(f"🧠 EP-Editor: {initial_response}")



# Main function for the chat loop
def main():
    """
    Main loop for the chat application.
    """
    print("Welcome to EP-Editor!")
    print("Check this out!")
    print("Chat History:")

    # Add AI's initial response
    initialize_chat()

    while True:
        user_input = input("👤 User: ").strip()

        if user_input.lower() == "exit":
            print("Exiting EP-Editor. Goodbye!")
            break
        elif user_input.lower() == "clear":
            chat_history.clear()
            initialize_chat()  # Reinitialize chat with the initial response
        else:
            # Add user input to chat history
            chat_history.append({"user": user_input, "assistant": None})

            # Prepare the full history as a single prompt text
            full_history = get_history()
            input_text = "\n".join(
                message.prompt.template for message in full_history
            )  # Combine all messages into a single prompt

            # Generate the response from the AI
            response = generate_response(input_text)

            # Save the AI's response to the chat history
            chat_history[-1]["assistant"] = response
            print(f"🧠 EP-Editor: {response}")

# Run the main loop
if __name__ == "__main__":
    main()
