from workflow import app, config, HumanMessage, ToolMessage
import logging
import pprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('workflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    while True:
        user_input = input("\nEnter your query (or 'exit' to quit): ")
        logger.info(f"Received user input: {user_input}")

        if user_input.lower() == 'exit':
            logger.info("User requested exit")
            print("Goodbye!")
            break

        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "outline": "",
            "images": [],
            "found_information": [],
            "input": user_input,
            "slides": [],
            "summary": []
        }
        logger.info("Created initial state")

        try:
            logger.info("Starting workflow execution")
            result = app.invoke(initial_state, config=config)
            logger.info("Workflow execution completed")
            logger.debug(f"Final result: {result}")

            print("\n[DEBUG] Full result:")
            pprint.pprint(result)

            for m in result["messages"]:
                if isinstance(m, ToolMessage):
                    logger.debug(f"Tool message: {m.content}")
                    print(f"ToolMessage: {m.content}")
        except Exception as e:
            logger.error(f"Error in workflow execution: {str(e)}", exc_info=True)
            print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
