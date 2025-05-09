import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class GPT_4o(AzureChatOpenAI):
    def __init__(self, **kwargs):
        super().__init__(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            model="gpt-4o-mini",
            api_version="2024-08-01-preview",
            **kwargs  
        )
        
class GPT_o3(AzureChatOpenAI):
    def __init__(self, **kwargs):
        super().__init__(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            model="o3-mini",
            api_version="2024-12-01-preview",
            **kwargs  
        )
        