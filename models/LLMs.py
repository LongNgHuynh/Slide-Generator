import os
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from anthropic import AnthropicBedrock
from langchain_aws import ChatBedrockConverse
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

class Gemini(ChatGoogleGenerativeAI):
    def __init__(self, **kwargs):
        super().__init__(
            model="gemini-2.5-pro-preview-05-06",
            **kwargs
        )

    
class Claude_3_7_Sonnet(ChatBedrockConverse):
    def __init__(self, **kwargs):
        super().__init__(
                model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION_NAME"),
                max_tokens=1024,
            **kwargs
        )