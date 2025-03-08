"""
Main entry point for the agent-to-agent chat demo.

This module initializes and runs the Agent Marketplace platform, which serves as
a two-sided marketplace connecting users with AI agents.
"""
import argparse
from textwrap import dedent

from agent_marketplace.marketplace import AgentMarketplace
from agent_marketplace.agents.personal_ai import PersonalAI
from agent_marketplace.agents.food_delivery_agent import FoodDeliveryAgent


def main():
    # Add argument parser
    parser = argparse.ArgumentParser(description='Agent Marketplace Demo')
    parser.add_argument('--user_name', type=str, default="Nicholas Richmond",
                       help='The name of the user')
    parser.add_argument('--user_intent', type=str, default="Please help me order some food.",
                       help='The intent/task that the user wants to accomplish')
    args = parser.parse_args()

    # 1. Initialize personal AI agent
    personal_ai = PersonalAI(
        name=f"{args.user_name}'s Personal AI",
        owner=args.user_name,
        description="A personal AI agent that can help with tasks and provide information",
        user_intent=args.user_intent,
        model_config={
            "model": "openai:gpt-4o-mini",
        }
    )

    # 2. Initialize food delivery agent
    service_agent = FoodDeliveryAgent(
        name="Byte AI Agent",
        owner="Byte AI",
        description="A food delivery agent that can help users order food from restaurants.",
        user_intent=args.user_intent,
        model_config={
            "model": "openai:gpt-4o-mini",
        }
    )

    # 3. Initialize agent marketplace
    agent_marketplace = AgentMarketplace()

    # 4. Add agents to marketplace
    agent_marketplace.add_agent(personal_ai)
    agent_marketplace.add_agent(service_agent)

    # 5. Start agent chat
    agent_marketplace.start_agent_chat(
        agent_name_1=personal_ai.name,
        agent_name_2=service_agent.name,
    )


if __name__ == "__main__":
    main()
