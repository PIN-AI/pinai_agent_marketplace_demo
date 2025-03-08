import os
from typing import Dict, List, Any, Optional, Union
from openai import OpenAI

from agent_marketplace.config import get_settings
from agent_marketplace.schemas.agents import Context

class OpenAILLMProvider:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.settings = get_settings()
        self.api_key = self.config.get("api_key") or self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = self.config.get("model", "gpt-4o")
        self.client = OpenAI(api_key=self.api_key)
        
    def generate(self, prompt: str, system_prompt: str = "", context: Context = None, 
                 tools: Optional[List[Dict[str, Any]]] = None, 
                 tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto") -> Dict[str, Any]:
        """
        Generate text using LLM with optional tool support
        
        Args:
            prompt (str): The user prompt to generate a response for
            system_prompt (str, optional): System prompt to guide model behavior
            context (Context, optional): Conversation history and context
            tools (List[Dict[str, Any]], optional): List of tools in OpenAI format for function calling
            tool_choice (Union[str, Dict[str, Any]], optional): Tool choice parameter - "auto", "none", or specific tool config
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - content (str): The generated text response
                - tool_calls (Optional[List]): Tool call information if tools were used
                
        Raises:
            ValueError: If API key is not provided or API call fails
        """
        if not self.api_key:
            raise ValueError("API key not provided")
            
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add context if provided
        if context:
            for msg in context.history:
                messages.append({"role": msg.role, "content": msg.content})
                
        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        # Call OpenAI API using the official client
        try:
            # Prepare API call parameters
            api_params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.config.get("temperature", 0.7),
                "max_tokens": self.config.get("max_tokens", 1000),
            }
            
            # Add tools and tool_choice if provided
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = tool_choice
            
            response = self.client.chat.completions.create(**api_params)
            
            return {
                "content": response.choices[0].message.content,
                "tool_calls": response.choices[0].message.tool_calls
            }
            
        except Exception as e:
            raise ValueError(f"OpenAI API error: {str(e)}") 
