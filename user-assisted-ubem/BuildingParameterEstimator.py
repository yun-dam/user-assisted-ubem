from langchain_core.prompts import HumanMessagePromptTemplate, AIMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from eppy.modeleditor import IDF

class BuildingParameterEstimator:
    def __init__(self, model_url="http://localhost:11434/", model_name="llama3.2:3b"):
        """Initialize the model and other required components."""
        self.model = ChatOllama(model=model_name, base_url=model_url)
        self.chat_history = []
        self._initialize_prompts()
        self._initialize_output_parser()

    def _initialize_prompts(self):
        """Initialize prompt templates."""
        self.estimate_template = """\
        For the following text about building information, your task is to estimate the hourly occupancy schedule of this building by asking the building user a series of questions. The occupancy schedule should be represented on a scale from 0 (vacant) to 1 (fully occupied), for example, 0.5 would be 50% occupied. Begin by asking a question. You may ask multiple questions in sequence if it helps refine your estimation. After each response and question, update your current estimation.

        current_estimation: Based on your current estimation, how is the hourly occupancy schedule of this building? Output them as a comma-separated Python list of 24 estimation samples from 0 hour to 23 hour. Ensure that the estimated values align with realistic occupancy patterns. 
        For example:
        - Occupancy is typically higher during business hours (e.g., 9 AM - 5 PM) for office buildings.
        - Residential buildings often show higher occupancy during evenings and early mornings.
        - Occupancy must remain between 0 and 1.
        If your current estimation significantly deviates from typical patterns, include a justification and explicitly state your reasoning.

        For an office building with business hours from 8 AM to 6 PM:
        current_estimation: [0, 0, 0, 0, 0, 0, 0.1, 0.8, 1, 1, 1, 1, 1, 1, 0.9, 0.7, 0.3, 0.1, 0, 0, 0, 0, 0, 0]

        following_question: If you need more information to estimate occupancy schedule more accurately, ask user relevant questions.

        validation_check: A brief explanation of how the estimation aligns with typical patterns, or why it deviates.

        Every time you respond, format the output in every conversation as JSON with the following keys:

          "current_estimation": [24 hourly values from 0 to 1],
          "following_question": "Your next question, or null if no further questions are needed",
          "validation_check": "A brief explanation of how the estimation aligns with typical patterns, or why it deviates."

        Ask these clarifying questions first:
        - "What are the typical hours of operation or occupancy?"
        - "Are there any specific times when the building is fully or minimally occupied?"

        building information: {text}

        {format_instructions}
        """
        self.first_template = """\
        Provide the first question now based on the provided building information. Format the output in the key "following question".
        """

    def _initialize_output_parser(self):
        """Initialize the structured output parser."""
        self.current_estimation_schema = ResponseSchema(
            name="current_estimation",
            description="Based on your current estimation, how is the hourly occupancy schedule of this building? Output them as a comma-separated Python list of 24 estimation samples from 0 hour to 23 hour."
        )
        self.following_question_schema = ResponseSchema(
            name="following_question",
            description="If you need more information to estimate occupancy schedule more accurately, ask user relevant questions. This includes the first question."
        )
        self.validation_check_schema = ResponseSchema(
            name="validation_check",
            description="A brief explanation of how the estimation aligns with typical patterns, or why it deviates."
        )
        self.output_parser = StructuredOutputParser.from_response_schemas([
            self.current_estimation_schema,
            self.following_question_schema,
            self.validation_check_schema
        ])
        self.format_instructions = self.output_parser.get_format_instructions()

    def create_estimation(self, building_info: str):
        """Generate the first estimation and question for a building."""
        system_prompt = SystemMessagePromptTemplate.from_template(template=self.estimate_template)
        system_messages = system_prompt.format_messages(
            text=building_info,
            format_instructions=self.format_instructions
        )
        first_prompt = HumanMessagePromptTemplate.from_template(template=self.first_template)
        first_messages = first_prompt.format_messages()

        messages = system_messages + first_messages
        response = self.model.invoke(messages)
        output_dict = self.output_parser.parse(response.content)

        # Save to chat history
        self.chat_history.extend(system_messages)
        return output_dict

    def refine_estimation(self, user_response: str):
        """Refine the estimation based on user responses with error handling."""
        try:
            # Append user input to chat history
            self.chat_history.append(HumanMessage(content=user_response))

            # Generate AI messages based on the latest user input
            ai_prompt = AIMessagePromptTemplate.from_template(template=self.chat_history[-1].content)
            ai_messages = ai_prompt.format_messages()
            messages_combined = self.chat_history + ai_messages

            # Invoke the model and parse the response
            response_combined = self.model.invoke(messages_combined)
            
            try:
                output_dict_combined = self.output_parser.parse(response_combined.content)
            except Exception as parse_error:
                raise ValueError(f"Error parsing model response: {response_combined.content}") from parse_error

            # Validate the parsed response contains required keys
            required_keys = {"current_estimation", "following_question", "validation_check"}
            if not all(key in output_dict_combined for key in required_keys):
                raise ValueError(f"Invalid response format. Missing keys in: {output_dict_combined}")

            # Append the new question to chat history and return the output
            self.chat_history.append(AIMessage(content=output_dict_combined.get("following_question")))
            return output_dict_combined

        except Exception as e:
            # Handle unexpected errors and provide fallback behavior
            print(f"An error occurred during refinement: {e}")
            return {
                "current_estimation": None,
                "following_question": None,
                "validation_check": f"An error occurred: {e}"
            }

            return output_dict_combined

    def revise_schedule_compact(self, idf_path: str, schedule_name: str, hourly_values: list):
        """
        Revises a Schedule:Compact object in the IDF file with the provided hourly values.

        Parameters:
        - idf_path: Path to the IDF file.
        - schedule_name: Name of the Schedule:Compact object to modify.
        - hourly_values: List of 24 float values representing hourly schedule.
        """
        # Validate input length
        print(hourly_values)
        if len(hourly_values) != 24:
            raise ValueError("Hourly values list must contain exactly 24 values.")

        # Load the IDF file
        IDF.setiddname("./ep-model/Energy+.idd")  
        idf = IDF(idf_path)

        # Find the Schedule:Compact object
        schedules = idf.idfobjects['SCHEDULE:COMPACT']
        for schedule in schedules:
            if schedule.Name == schedule_name:
                # Generate the "Until" time fields
                until_times = [f"{i + 1:02}:00" for i in range(24)]

                # Update the fields in Schedule:Compact
                for i in range(24):
                    until_field = f"Field_{3 + (i * 2)}"  # Alternating "Until" fields
                    value_field = f"Field_{4 + (i * 2)}"  # Corresponding value fields
                    setattr(schedule, until_field, until_times[i])  # Set "Until" time
                    setattr(schedule, value_field, hourly_values[i])  # Set value

                print(f"Updated Schedule:Compact {schedule_name} with new hourly values.")
                break
        else:
            print(f"Schedule:Compact with name '{schedule_name}' not found.")
            return

        # Save the updated IDF
        updated_idf_path = "./ep-model/updated_ep.idf"
        idf.saveas(updated_idf_path)
        print(f"Updated IDF file saved at {updated_idf_path}")


if __name__ == "__main__":
    estimator = BuildingParameterEstimator()
    building_info = """Building name: Gates building\nYear of construction: 2011\nBuilding usetype: Education"""

    # Generate initial estimation and retrieve the first question
    initial_output = estimator.create_estimation(building_info)
    print("Initial Estimation Output:", initial_output)

    # Start refining the estimation
    refined_output = initial_output  # Start with initial output
    while True:
        # Display the appropriate question
        question = refined_output.get("following_question")
        if not question:
            print("\nNo further questions. Refinement complete.")
            break

        user_response = input(f"\n{question} (or type 'done' to finish):\n")
        if user_response.lower() == 'done':
            break

        # Refine the estimation based on user response
        refined_output = estimator.refine_estimation(user_response)
        print("\nRefined Estimation Output:\n", refined_output)

    # Update IDF Schedule:Compact
    hourly_schedule = refined_output.get("current_estimation")
    estimator.revise_schedule_compact("./ep-model/medium_doe_office_ehp.idf", "BLDG_OCC_SCH", hourly_schedule)

