# Agent Marketplace Demo

A demonstration of the PIN AI agent marketplace that enables the creation of personalized AI assistants based on user data and service agents that fulfill real-world tasks. Personal AI agents can interact with service agents on behalf of users to seamlessly complete various tasks.

**PIN AI and Byte AI Food Delivery Agent Interaction**

<video src="https://github.com/user-attachments/assets/a610617e-7653-41bf-8537-d932ded00263" width="80%" controls></video>

## ⚡ Installation

#### 1. Clone the repository
```bash
git clone https://github.com/PIN-AI/pinai_agent_marketplace_demo
```
#### 2. Install dependencies
```
cd agent_marketplace
pip install -e .
```

#### 3. Create .env
```
cp .env.example .env
```
Fill in values in `.env`

## 🤖 Start Agent-to-agent Chat
```
streamlit run examples/agent_to_agent_chat.py  
```
(Always have one streamlit window running to avoid unexpected issues)

## ⚙️ Integrate with Your Own Agent

#### 1. Create an agent under `agent_marketplace/agents/my_agent.py`
```
from agent_marketplace.agents.ai_agent import AI_Agent
from agent_marketplace.schemas.agents import Message


class MyAwesomeAgent(AI_Agent):
    def __init__(self, name: str, owner: str, description: str, model_config: dict = {}):
        super().__init__(name, owner, description, model_config)

    def on_message(self, message: Message, sender: AI_Agent) -> Message:
        """What to reply when receives a message"""
    	pass
```

#### 2. Add your agent to the agent marketplace

Replace the code snippet [examples/agent_to_agent_chat.py](https://github.com/PIN-AI/pinai_agent_marketplace_demo/blob/main/examples/agent_to_agent_chat.py#L34-L43) with:

```
# 2. Initialize your agent
service_agent = MyAwesomeAgent(
    name="My awesome agent",
    owner="Foodie Inc.",
    description="Write a description about your agent, as detailed as possible.",
    user_intent=args.user_intent,
    model_config={}
)
```

#### 3. Run the streamlit command [above](https://github.com/PIN-AI/pinai_agent_marketplace_demo?tab=readme-ov-file#-start-agent-to-agent-chat) to start agent-to-agent chat.

## 🧠 Improve Personal AI

You are encouraged to modify the [personal AI agent](https://github.com/PIN-AI/pinai_agent_marketplace_demo/blob/main/agent_marketplace/agents/personal_ai.py) as well to make it work better with your agent! 