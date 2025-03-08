from agent_marketplace.schemas.agents import Context, Message

class AI_Agent:
    def __init__(self, name: str, owner: str, description: str, model_config: dict = {}):
        self.name = name
        self.owner = owner
        self.description = description
        self.model_config = model_config
        self.context = Context(history=[])
        self.task_complete = False

    def init_chat(self, guest_agent: "AI_Agent" = None):
        pass
    
    def on_message(self, message: Message, sender: "AI_Agent") -> str:
        raise NotImplementedError("Subclasses must implement this method")
