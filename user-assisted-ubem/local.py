from langchain_core.prompts import HumanMessagePromptTemplate, AIMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Initialize the model
model = ChatOllama(model='llama3.2:3b', base_url="http://localhost:11434/")

estimate_template = """\
For the following text about building information, your task is to estimate the hourly occupancy schedule of this building by asking the building user a series of questions. The occupancy schedule should be represented on a scale from 0 (vacant) to 1 (fully occupied). Begin by asking a question. You may ask multiple questions in sequence if it helps refine your estimation. After each response and question, update your current estimation"

current_estimation: Based on your current estimation, how is the hourly occupancy schedule of this building? Output them as a comma separated Python list of 24 estimation samples from 0 hour to 23 hour.

following_question: If you need more information to estimate occupancy schuedule more accurately, ask user relevant questions.

Everytime you response, format the output in every conversation as JSON with the following keys:

current_estimation
following_question

building information: {text}

{format_instructions}
"""

building_information_template = """\
Building name: Gates building
Year of construction: 2011
Building usetype: Education
"""

first_template = """\

Provide the first question now based on the provided building information. Format the output in the key "following qeusiton

"""

# Prompt templates and output parser setup
current_estimation_schema = ResponseSchema(
    name="current_estimation",
    description="Based on your current estimation, how is the hourly occupancy schedule of this building? Output them as a comma separated Python list of 24 estimation samples from 0 hour to 23 hour."
)

following_question_schema = ResponseSchema(
    name="following_question",
    description="If you need more information to estimate occupancy schedule more accurately, ask user relevant questions. This includes the first question."
)

response_schemas = [current_estimation_schema, following_question_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

system_prompt = SystemMessagePromptTemplate.from_template(template=estimate_template)

system_messages = system_prompt.format_messages(
    text=building_information_template,
    format_instructions=format_instructions
)

first_prompt = HumanMessagePromptTemplate.from_template(template=first_template)
first_messages = first_prompt.format_messages()
    
messages = system_messages + first_messages

response = model.invoke(messages)
output_dict = output_parser.parse(response.content)

# Chat history initialization
chat_history = []

chat_history.append(system_messages)
print(f"🧠 EP-Editor: {output_dict.get("following_question")}")

user_input = "First class starts at 8:30am and most of researchers and professors come to the building around 9 am and leave around 5 pm."
chat_history.append(HumanMessagePromptTemplate.from_template(template=user_input))

ai_prompt =  AIMessagePromptTemplate.from_template(template=output_dict.get("following_question"))
ai_messages = ai_prompt.format_messages()

user_prompt = HumanMessagePromptTemplate.from_template(template=user_input)
user_messages = user_prompt.format_messages()

messages2 = system_messages + ai_messages + user_messages

response2 = model.invoke(messages2)
output_dict2 = output_parser.parse(response2.content)
