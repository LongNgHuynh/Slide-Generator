import os
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# from anthropic import AnthropicBedrock
# from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
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
            model="gemini-2.5-pro",
            **kwargs
        )

class Gemini_2_5_Flash(ChatGoogleGenerativeAI):
    def __init__(self, **kwargs):
        super().__init__(
            model="gemini-2.5-flash",
            **kwargs
        )

    
class Claude_3_7_Sonnet(ChatBedrock):
    def __init__(self, **kwargs):
        super().__init__(
                model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION_NAME"),
                max_tokens=200000,
            **kwargs
        )
        
if __name__ == "__main__":
    llm = Claude_3_7_Sonnet()
    print(llm.invoke("Hello, how are you?"))