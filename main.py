from models.LLMs import Claude_3_7_Sonnet

llm = Claude_3_7_Sonnet()

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg)