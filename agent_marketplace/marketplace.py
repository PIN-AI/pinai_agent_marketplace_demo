from datetime import datetime

import streamlit as st

from agent_marketplace.agents.ai_agent import AI_Agent
from agent_marketplace.schemas.agents import Message
from agent_marketplace.config import response_generator, setup_streamlit

class AgentMarketplace:
    def __init__(self):
        self.agents: dict[str, AI_Agent] = {}
        self.max_chat_round = 40  # Maximum number of rounds for the chat
        
    def add_agent(self, agent: AI_Agent) -> None:
        self.agents[agent.name] = agent
    
    def remove_agent(self, agent_name: str) -> None:
        del self.agents[agent_name]
    
    def get_agent(self, agent_name: str) -> AI_Agent:
        return self.agents[agent_name]
    
    def list_agents(self) -> list[str]:
        return list(self.agents.keys())
    
    def start_agent_chat(self, agent_name_1: str, agent_name_2: str):
        print("\n" + "="*80)
        print("🤖 Agent-to-Agent Chat 🤖".center(80))
        print("="*80 + "\n")

        # Setup streamlit
        setup_streamlit()

        # Create agents
        agent_1 = self.agents[agent_name_1]
        agent_2 = self.agents[agent_name_2]

        print("\n" + "-"*80)
        print(f"Creating agents: \033[1;34m{agent_1.name}\033[0m and \033[1;32m{agent_2.name}\033[0m")
        print("-"*80 + "\n")
        with st.chat_message("user"):
            st.write_stream(
            response_generator(f"Hi :blue[**{agent_1.owner}**], I am your personal AI and I am here to help you with your task.")
            )
            st.write_stream(
                response_generator(f'📝 Task: :blue[**"{agent_1.user_intent}"**]')
            )
            st.write_stream(
                response_generator(f"🤖 Service Agent: :violet[**{agent_2.name}**]")
            )

        # Initialize chat
        print("\n" + "-"*80)
        print(f"Initializing chat for \033[1;34m{agent_1.name}\033[0m and \033[1;32m{agent_2.name}\033[0m")
        print("-"*80 + "\n")
        agent_1.init_chat(guest_agent=agent_2)
        agent_2.init_chat(guest_agent=agent_1)

        # Agent 1 initiate the conversation
        sender = agent_1
        sender_message = agent_1.on_message(  # Agent 1 generates the first message
            Message(role="user", content="", sender=agent_1.name, receiver=agent_2.name, timestamp=datetime.now()), 
            sender=agent_2
        )  
        print(f"\033[1;34m{agent_1.name}\033[0m:\n\n{sender_message.content}") 
        print("\n" + "-"*100 + "\n")
        st.chat_message("user").write_stream(
                response_generator(f"**[💬 Start to chat with 🤖 :violet[{agent_2.name}]]**")
            )
        st.chat_message("user").write_stream(response_generator(sender_message.content))
                
        # Conversation loop starts
        receiver = agent_2
        round = 0
        while round < self.max_chat_round:
            response = receiver.on_message(sender_message, sender=sender)
            if receiver == agent_1:
                print(f"\033[1;34m{receiver.name}\033[0m:\n\n{response.content}")
                print("\n" + "-"*100 + "\n")
                st.chat_message("user").write_stream(response_generator(response.content))
            else:
                print(f"\033[1;32m{receiver.name}\033[0m:\n\n{response.content}")
                print("\n" + "-"*100 + "\n")
                st.chat_message("assistant").write_stream(response_generator(response.content))
            
            sender = agent_1 if sender == agent_2 else agent_2
            sender_message = response

            receiver = agent_1 if sender == agent_2 else agent_2

            if agent_1.task_complete and agent_2.task_complete:
                break

            round += 1
        
        if round >= self.max_chat_round:
            print("Communication is not completed within the maximum number of rounds.")
        else:
            print("Communication ends successfully.")
