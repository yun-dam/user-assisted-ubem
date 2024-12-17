from langchain_core.prompts import HumanMessagePromptTemplate, AIMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

# Initialize the model
model = ChatOllama(model='llama3.2:3b', base_url="http://localhost:11434/")

estimate_template = """\
For the following text about building information, your task is to estimate the hourly occupancy schedule of this building by asking the building user a series of questions. The occupancy schedule should be represented on a scale from 0 (vacant) to 1 (fully occupied). Begin by asking a question. You may ask multiple questions in sequence if it helps refine your estimation. After each response and question, update your current estimation"

current_estimation: Based on your current estimation, how is the hourly occupancy schedule of this building? Output them as a comma separated Python list of 24 estimation samples from 0 hour to 23 hour.

following_question: If you need more information to estimate occupancy schuedule more accurately, ask user relevant questions. 

Format the output as JSON with the following keys:

current_estimation
following_question

text: {text}

{format_instructions}
"""

building_information_template = """\
Building name: Gates building
Year of construction: 2011
Building usetype: Education
"""

# Prompt templates and output parser setup
current_estimation_schema = ResponseSchema(
    name="current_estimation",
    description="Based on your current estimation, how is the hourly occupancy schedule of this building? Output them as a comma separated Python list of 24 estimation samples from 0 hour to 23 hour."
)

following_question_schema = ResponseSchema(
    name="following_question",
    description="If you need more information to estimate occupancy schedule more accurately, ask user relevant questions."
)

response_schemas = [current_estimation_schema, following_question_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

prompt = ChatPromptTemplate.from_template(template=estimate_template)
messages = prompt.format_messages(
    text=building_information_template,
    format_instructions=format_instructions
)

response = model.invoke(messages)
output_dict = output_parser.parse(response.content)

# Chat history initialization
chat_history = []

chat_history.append({"user": None, "assistant": output_dict})
print(f"🧠 EP-Editor: {output_dict.get("following_question")}")

user_input = "200 students"
chat_history.append({"user": user_input, "assistant": None})



history = [prompt]



for chat in chat_history:
    if chat['user']:  # Add user messages
        prompt_user = HumanMessagePromptTemplate.from_template(chat['user'])
        history.append(prompt_user)

    if chat['assistant']:  # Add assistant responses
        ai_message = AIMessagePromptTemplate.from_template(chat['assistant'])
        history.append(ai_message)

### 12/15/2024 5pm output_dict에서 history 쌓아서 계속 대화 이어가는 방향으로 코딩 ###
# Function to initialize chat with the AI
def initialize_chat():
    """
    Initializes the chat by generating the first response to the system message.
    """
    # Generate the initial response from the AI
    response = model.invoke(messages)
    initial_response = output_parser.parse(response.content)

    # Add the system message and AI's response to chat history
    chat_history.append({"user": None, "assistant": initial_response})
    print(f"🧠 EP-Editor: {initial_response.get("following_question")}")


# Function to prepare the full chat history
def get_history():
    """
    Combines the system message and the chat history into a single structured list.
    """
    history = [prompt]

    for chat in chat_history:
        if chat['user']:  # Add user messages
            prompt_user = HumanMessagePromptTemplate.from_template(chat['user'])
            history.append(prompt)

        if chat['assistant']:  # Add assistant responses
            ai_message = AIMessagePromptTemplate.from_template(chat['assistant'])
            history.append(ai_message)

    return history


def main():
    """
    Main loop for the chat application with iterative improvement for estimation.
    """
    print("Welcome to the EP-Editor!")
    print("Type 'exit' to quit or 'clear' to reset the chat.")

    # Add AI's initial response
    initialize_chat()

    while True:
        # Check for AI's "following_question" in the last response to guide user input
        if chat_history and chat_history[-1]["assistant"]:
            assistant_response = chat_history[-1]["assistant"]
            following_question = assistant_response.get("following_question", None)
            if following_question:
                print(f"🧠 EP-Editor: {following_question}")

        # Get user input
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
            response = model.invoke(input_text)
            parsed_response = output_parser.parse(response.content)

            # Save the AI's response to the chat history
            chat_history[-1]["assistant"] = parsed_response

            # Display the AI's response
            current_estimation = parsed_response.get("current_estimation", None)
            following_question = parsed_response.get("following_question", None)

            if current_estimation:
                print(f"📊 Current Estimation: {current_estimation}")
            if following_question:
                print(f"🧠 EP-Editor: {following_question}")


# Run the main loop
if __name__ == "__main__":
    main()
