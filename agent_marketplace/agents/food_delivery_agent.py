import requests
import json
from datetime import datetime

from agent_marketplace.agents.ai_agent import AI_Agent
from agent_marketplace.services.llm import OpenAILLMProvider
from agent_marketplace.agents.personal_ai import CHECK_CHAT_STATE_PROMPT
from agent_marketplace.schemas.agents import Message
from agent_marketplace.services.geocoding import get_coordinates_from_address
from agent_marketplace.config import get_settings

SERVER_URL = "http://54.70.105.247:8000"


class FoodDeliveryAgent(AI_Agent):
    def __init__(self, name: str, owner: str, description: str, user_intent: str, model_config: dict = {}):
        super().__init__(name, owner, description, model_config)
        self.chat_id = None
        self.user_intent: str = user_intent

        self.user_info = {
            "user_address": "",
            "user_name": "",
            "user_phone_number": "",
            "wallet_address": "0xac8CeB5131449e5850030737D10d1E25C6b8D80B",
            "flow": "regular",
        }

    def generate_chat_id(self):
        response = requests.post(
            f"{SERVER_URL}/init_chat",
            json={
                "user_address": self.user_info["user_address"],
                "user_name": self.user_info["user_name"],
                "user_phone_number": self.user_info["user_phone_number"],
                "wallet_address": self.user_info["wallet_address"],
                "flow": self.user_info["flow"],
                "user_lat": self.user_info["user_lat"],
                "user_lon": self.user_info["user_lon"]
            },
        )
    
        if response.status_code != 200:
            print("Failed to start chatbot. Exiting.")
            print(response)
            return
    
        self.chat_id = response.json().get("chat_id")

    def on_message(self, message: Message, sender: AI_Agent) -> Message:
        # Update context
        if message:
            self.context.history.append(message)

        # Generate response
        response = self.generate_response(message, sender)
        return_message = Message(role="assistant", content=response["text"], sender=self.name, receiver=sender.name, 
                                 timestamp=datetime.now(), metadata=response if "paymentDetails" in response else {})

        # Update context
        self.context.history.append(return_message)

        # Check if the task is complete
        if response == "[CONVERSATION_ENDS]":
            self.task_complete = True

        return return_message
    
    def generate_response(self, message: Message, sender: AI_Agent) -> str:
        """
        Generate response for the incoming message.
        """
        # Request user info if any required user info is missing
        response = self.request_user_info(message)
        if response:
            return response

        # Check if the chat should end at this turn
        chat_state = self.llm_call_to_check_chat_state()
        if chat_state["content"] == "[CONVERSATION_ENDS]":
            self.task_complete = True
            return {"text": "[CONVERSATION_ENDS]"}

        # Call Byte AI API to get response
        chat_response = requests.post(
            f"{SERVER_URL}/send_message/{self.chat_id}",
            json={"message": message.content},
        )
        if chat_response.status_code != 200:
            print("Error communicating with chatbot:", chat_response.json().get("error", "Unknown error"))
            raise Exception("Error communicating with chatbot")
        bot_reply = json.loads(chat_response.text)["response"]
        bot_reply["text"] = bot_reply["text"].strip()

        return bot_reply

    def request_user_info(self, message: Message) -> dict:
        if not all([
            self.user_info["user_address"],
            self.user_info["user_name"], 
            self.user_info["user_phone_number"],
        ]):
            try:
                msg_json = json.loads(message.content)
                self.user_info["user_name"] = msg_json["user_name"]
                self.user_info["user_phone_number"] = msg_json["user_phone_number"]

                # Get coordinates from address
                coordinates = get_coordinates_from_address(msg_json["user_address"])
                self.user_info["user_address"] = coordinates["formatted_address"]
                self.user_info["user_lat"] = coordinates["lat"]
                self.user_info["user_lon"] = coordinates["lng"]
                
                # Generate chat id
                self.generate_chat_id()

                return {"text": "Thank you very much for providing your information! What can I do for you today?"}
            except Exception as e:
                print(f"Error requesting user info: {e}")
                return {
                    "text": 'My pleasure to serve you! Could you please provide your name, phone number and address? In the \
                            following JSON format only:\n\n {"user_name": "<client_name>", "user_phone_number": "<client_phone_number>", \
                            "user_address": "<client_address>"}'
                }
        return {}
            
    def llm_call_to_check_chat_state(self) -> dict:
        """
        Check if the agent should complete the chat at this turn. If to complete chat, return "[CONVERSATION_ENDS]", otherwise return "[CONTINUE]".
        """
        conversation_history = "\n".join([f"{msg.sender}: {msg.content}" for msg in self.context.history][-10:])

        if "[PAYMENT_SUCCEEDED]" in conversation_history or "[CONVERSATION_ENDS]" in conversation_history:
            return {"content": "[CONVERSATION_ENDS]"}

        prompt = CHECK_CHAT_STATE_PROMPT.format(
            agent_name=self.name,
            owner=self.owner,
            user_intent=self.user_intent,
            service_agent_description=self.description,
            conversation_history=conversation_history,
        )

        llm_call = OpenAILLMProvider()
        response = llm_call.generate(prompt=prompt)
        return response
