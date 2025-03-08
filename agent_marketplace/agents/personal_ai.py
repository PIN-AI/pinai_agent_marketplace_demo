import os
import json
from datetime import datetime
from textwrap import dedent
import streamlit as st

from agent_marketplace.agents.ai_agent import AI_Agent
from agent_marketplace.config import get_settings
from agent_marketplace.schemas.agents import Message
from agent_marketplace.services.llm import OpenAILLMProvider
from agent_marketplace.tools import registered_tools
from agent_marketplace.config import response_generator

class PersonalAI(AI_Agent):
    def __init__(self, name: str, owner: str, description: str, user_intent: str, model_config: dict = {}):
        super().__init__(name, owner, description, model_config)
        self.settings = get_settings()
        self.personal_basic_info: str = ""
        self.personal_preferences: dict[str, str] = {}
        self.user_intent: str = user_intent

        self.llm = OpenAILLMProvider()
        self.tools = registered_tools

    def init_chat(self, guest_agent: AI_Agent = None):
        # Retrieve personal information based on the guest agent
        self.retrieve_personal_preferences(guest_agent)

    def on_message(self, message: Message, sender: AI_Agent) -> Message:
        # Update context
        if message.content:
            self.context.history.append(message)

        # Generate response
        response = self.generate_response(message, sender)

        # Update context
        if response:
            self.context.history.append(Message(role="user", content=response["content"], sender=self.name, receiver=sender.name, timestamp=datetime.now()))

        # Check if the task is complete
        if response == "[CONVERSATION_ENDS]":
            self.task_complete = True

        return Message(role="user", content=response["content"], sender=self.name, receiver=sender.name, timestamp=datetime.now())

    def retrieve_personal_preferences(self, sender: AI_Agent) -> str:
        print(f"Retrieving personal preferences for \033[1;33m{self.owner}\033[0m")
        with st.chat_message("user"):
            st.write_stream(
                response_generator(f"🔍 **Retrieving personal preferences for :blue[{self.owner}]**")
            )

            personal_data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "personal_data", self.owner)
            if not os.path.exists(personal_data_dir):
                raise ValueError(f"Personal data directory {personal_data_dir} does not exist. Is the client name correct?")

            # Get basic info
            self.personal_basic_info = self.llm_call_to_summarize_personal_preferences(sender, open(os.path.join(personal_data_dir, "basic_info.json")).read())

            # Get personal preferences
            personal_preferences = []
            for file in os.listdir(personal_data_dir):
                if file.endswith(".json") and file != "basic_info.json":
                    personal_data = json.load(open(os.path.join(personal_data_dir, file)))
                    p_info = self.llm_call_to_retrieve_personal_info(sender, personal_data)["content"]
                    st.write_stream(
                        response_generator(f"Searching in **{os.path.splitext(file)[0]}** ...")
                    )
                    st.write_stream(
                        response_generator(p_info)
                    )
                    personal_preferences.append(p_info)
        
        with st.chat_message("user"):
            # Summarize personal preferences
            personal_preferences = self.llm_call_to_summarize_personal_preferences(sender, "\n\n".join(personal_preferences))
            self.personal_preferences[sender.name] = personal_preferences["content"]
            st.write_stream(
                response_generator(f"✅ **Summarizing :blue[{self.owner}]'s personal preferences**")
            )
            st.write_stream(
                response_generator(self.personal_preferences[sender.name])
            )

            # print(self.personal_preferences[sender.name])
            # exit()
        # return dedent("""
        #     Kelly Simon is a 35 year old male who likes to play basketball and code. 
        #     He likes spicy keba.
        #     Boliang lives in 8 Green Ridge Pl, Saratoga Springs, NY 12866
        #     His phone number is +12345678901
        # """)

    def generate_response(self, message: Message, sender: AI_Agent) -> str:
        # Check if the task is complete
        chat_state = self.llm_call_to_check_chat_state(sender)
        if chat_state["content"] == "[CONVERSATION_ENDS]":
            self.task_complete = True
            return {"content": "[CONVERSATION_ENDS]"}

        retry = 0
        validator_response = {}
        while retry < 3:
            # Generate response
            response = self.llm_call_to_generate_response(sender, validator_response)

            # Tool call
            if response["tool_calls"]:
                for tool_call in response["tool_calls"]:
                    tool_func_name = tool_call.function.name
                    tool_func_args = tool_call.function.arguments
                    tool_func_result = self.tools[tool_func_name](tool_func_args, message)
                    
                    # Update context
                    self.context.history.append(
                        Message(role="user", 
                                content=f"{tool_func_result}", 
                                sender=f"[{tool_func_name}] tool", 
                                receiver=sender.name,
                                timestamp=datetime.now())
                    )
                
                # Regenerate response
                response = self.llm_call_to_generate_response(sender, validator_response)
            
            # Validate response
            validator_response = self.llm_call_to_validate_response(response["content"], sender)
            if validator_response["content"] == "[YES]":
                return response
            
            # Remove the last message from the context
            self.context.history.pop()
            
            retry += 1

        return response

    def llm_call_to_validate_response(self, input_message: str, sender: AI_Agent) -> dict:
        prompt = VALIDATE_RESPONSE_PROMPT.format(
            owner=self.owner,
            user_intent=self.user_intent,
            service_agent_description=sender.description,
            conversation_history="\n".join([f"{msg.sender}: {msg.content}" for msg in self.context.history][-10:]),
            input_message=input_message
        )
        response = self.llm.generate(prompt=prompt)
        if response["content"] != "[YES]":
            response["content"] = f"# Notes\nPlease do not generate response like this: \n{input_message}\n\nThe reason is: \n{response['content']}"
        return response
    
    def llm_call_to_check_chat_state(self, sender: AI_Agent) -> dict:
        conversation_history = "\n".join([f"{msg.sender}: {msg.content}" for msg in self.context.history][-10:])
        
        if "[PAYMENT_SUCCEEDED]" in conversation_history:
            return {"content": "[CONVERSATION_ENDS]"}
        
        prompt = CHECK_CHAT_STATE_PROMPT.format(
            agent_name=self.name,
            owner=self.owner,
            user_intent=self.user_intent,
            service_agent_description=sender.description,
            conversation_history=conversation_history,
        )
        response = self.llm.generate(prompt=prompt)
        return response
    
    def llm_call_to_generate_response(self, sender: AI_Agent, validator_response: dict = {}) -> dict:
        # Generate response
        prompt = GENERATE_RESPONSE_PROMPT.format(
            owner=self.owner,
            owner_personal_info=f"{self.personal_basic_info}\n\n{self.personal_preferences[sender.name]}",
            service_agent_name=sender.name,
            service_agent_description=sender.description,
            conversation_history="\n".join([f"{msg.sender}: {msg.content}" for msg in self.context.history][-10:]),
            user_intent=self.user_intent,
            validator_message=validator_response["content"] if validator_response else "",
            self_name=self.name
        )
        response = self.llm.generate(prompt=prompt, tools=PERSONAL_AI_TOOLS)
        return response

    def llm_call_to_retrieve_personal_info(self, sender: AI_Agent, owner_personal_data: str) -> dict:
        prompt = RETRIEVE_PERSONAL_INFO_PROMPT.format(
            owner=self.owner,
            service_agent_name=sender.name,
            service_agent_description=sender.description,
            user_intent=self.user_intent,
            owner_personal_data=owner_personal_data
        )
        response = self.llm.generate(prompt=prompt)
        return response

    def llm_call_to_summarize_personal_preferences(self, sender: AI_Agent, owner_personal_data: str) -> dict:
        prompt = SUMMARIZE_PERSONAL_PREFERENCES_PROMPT.format(
            owner_personal_data=owner_personal_data
        )
        response = self.llm.generate(prompt=prompt)
        return response


# Prompt templates for Personal AI
GENERATE_RESPONSE_PROMPT = """
You are {self_name}. You fulfill task on behalf of {owner}.
You will be chatting with a service agent {service_agent_name} to complete a task.
You are provided the {owner}'s personal information and the description of the service agent.

# Task the user wants to complete
{user_intent}

# {owner}'s personal information
{owner_personal_info}

# Service agent's description
{service_agent_description}

# Conversation history
{conversation_history}

{validator_message}

# Task for you
You are {self_name} to generate a response to {service_agent_name}. Do not include {self_name} at the beginning of your response. If you find it difficult to complete the task after a few attempts, end the conversation politely.
"""

CHECK_CHAT_STATE_PROMPT = """
Here is a conversation history between {owner}'s personal AI agent and the service agent.
The conversation is about to complete a task.

# Task that {owner} wants to complete:
{user_intent}

# Description of the service agent:
{service_agent_description}

# Conversation history

{conversation_history}

# Task for you
You are {agent_name} to continue the conversation. Based on the conversation history above, please determine the state of the conversation.

Here are states you can choose from:
[CONTINUE]: The conversation is still in progress and the task is not completed. Participants are still discussing the task.
[CONVERSATION_ENDS]: The task is completed. Participants in the conversation are satisfied with the outcome. [CONVERSATION_ENDS] exists in the conversation history. [PAYMENT_SUCCEEDED] exists in the conversation history.

Only reply with one of the states above.
"""

VALIDATE_RESPONSE_PROMPT = """
Here is a conversation history between {owner}'s personal AI agent and the service agent.
The conversation is about to complete a task.

# Task that {owner} wants to complete:
{user_intent}

# Description of the service agent:
{service_agent_description}

# Conversation history
{conversation_history}

# Task for you
Now the personal AI is about to reply to the service agent with this response:
{input_message}

If you think the response is appropriate, please only reply with [YES].

If you think the response is not appropriate, please reply with one paragraph to explain why the response is not appropriate.
"""

RETRIEVE_PERSONAL_INFO_PROMPT = """
I am doing a task for my client {owner}. The agent I am working with is {service_agent_name}.

# The task description from my client
{user_intent}

# The description of the service agent
{service_agent_description}

# Your task
Search through {owner}'s personal information and select the information that is relevant to the task and the service agent.
Based on the selected personal info, estimate the personal information for the task and service agent. Generate one single paragraph of 50 words.

# The personal information of my client

{owner_personal_data}
"""

SUMMARIZE_PERSONAL_PREFERENCES_PROMPT = """
Please summarize the following personal information into one single paragraph of 100 words.

# Personal information
{owner_personal_data}
"""

# Personal AI tools descriptions
PERSONAL_AI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "process_coinbase_payment",
            "description": "Process a Coinbase Commerce payment. User this tool only if the user needs to pay for a coinbase commerce payment link."
        }
    },
]