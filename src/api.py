from fastapi import FastAPI
from pydantic import BaseModel
from crewai_fastapi_agent_chat.crew import EmailRagAgent

import os
import logging
import warnings
import traceback
import sqlite3
from datetime import datetime

# Suppress warnings and logs for cleaner output
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
logging.basicConfig(level=logging.ERROR)
logging.getLogger('opentelemetry').setLevel(logging.ERROR)
os.environ['OTEL_SDK_DISABLED'] = 'true'  # Disable telemetry if not required
app = FastAPI()

class CrewRequest(BaseModel):
    question: str
    # user_id: str
    username: str
    collection_id: str
    email: str
    mobilenumber: str
    chat_history : list
    prompt: str
    vector_db: str
    productId:list
    productname: list

@app.get('/check')
async def check():
    return "Server is running crewai bot..."



# Database setup for chat history
def initialize_db():
    """Initialize the SQLite database for storing chat history"""
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()

def store_chat_history(user_id, messages):
    """Store chat messages in the database"""
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    
    for message in messages:
        cursor.execute(
            'INSERT INTO chat_history (user_id, timestamp, role, content) VALUES (?, ?, ?, ?)',
            (user_id, timestamp, message['role'], message['content'])
        )
    
    conn.commit()
    conn.close()

def get_chat_history(user_id, limit=10):
    """Retrieve recent chat history for a specific user"""
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    
    cursor.execute(
        'SELECT role, content FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?',
        (user_id, limit*2)  # Multiply by 2 to get pairs of messages
    )
    
    results = cursor.fetchall()
    conn.close()
    
    # Convert to the format expected by the LLM
    history = []
    for role, content in results:
        history.append({"role": role, "content": content})
    
    # Reverse to get chronological order
    history.reverse()
    
    return history

def run_crew():
    try:
        initialize_db()
        while True:
            question = input("Enter your question (type 'q' to exit): ")
            if question.lower() == 'q':
                print("Exiting the chat. Goodbye!")
                break

            username = "anto"
            collection_id = "3"
            email = ""
            mobilenumber = "7010329187"
            vector_db = "staging"
            productId = ["1"]
            productname = ["DMS"]

            # Retrieve past conversations
            past_conversations = get_chat_history(mobilenumber)

            # Default prompt
            # prompt = """
            #     # System Message
            #     You are an AI sales assistant for Tabtree IT Consulting Services, specializing exclusively in Paperless Office AI-Based Enterprise DMS (Document Management System). Your role is to assist potential customers by answering questions strictly related to our DMS solutions, understanding their needs, and guiding them toward a decision.

            #     # Context
            #     {context}

            #     # Instructions
            #     1. Greet the customer warmly and introduce yourself as an AI assistant.
            #     2. Ask open-ended questions to understand the customer's needs and situation.
            #     3. Listen actively and tailor your responses to the customer's specific concerns.
            #     4. Highlight relevant features and benefits based on the customer's needs.
            #     5. Address any objections or concerns professionally and empathetically.
            #     6. If appropriate, guide the customer towards making a purchase or scheduling a demo.
            #     7. If the customer isn't ready to buy, offer additional resources or follow-up options.
            #     8. Always maintain a positive, helpful tone throughout the conversation.

            #     # Conversation History
            #     {chat_history}
            # """
            
            prompt = f"""
            # System Message
            You are an AI sales assistant for Tabtree IT Consulting Services, specializing exclusively in Paperless Office AI-Based Enterprise DMS (Document Management System). Your role is to assist potential customers by answering questions strictly related to our DMS solutions, understanding their needs, and guiding them toward a decision.

            # Context
            The following context may include company information, current promotions, or any other relevant background:
            {{context}}

            # Instructions
            - Greet the customer warmly and introduce yourself as an AI assistant.
            - Ask open-ended questions to understand the customer's needs and situation.
            - Listen actively and tailor your responses to the customer's specific concerns.
            - Highlight relevant features and benefits based on the customer's needs.
            - Address any objections or concerns professionally and empathetically.
            - If appropriate, guide the customer towards making a purchase or scheduling a demo.
            - If the customer isn't ready to buy, offer additional resources or follow-up options.
            - Always maintain a positive, helpful tone throughout the conversation.

            # Conversation History
            Below is the history of the conversation between you and the customer. Use it to craft your next response:
            {{chat_history}}
            """


            # Prepare input data
            input_data = {
                "question": question,
                "username": username,
                "collection_id": collection_id,
                "chat_history": past_conversations,
                "email": email,
                "prompt": prompt,
                "mobilenumber": mobilenumber,
                "product_id": productId,
                "product_name": productname,
                "vector_db": vector_db,
            }

            # Execute the agent
            result = EmailRagAgent(user_id=mobilenumber).crew().kickoff(inputs=input_data)

            # Extract the final task output
            final_output = result.raw
            print("Final Response:", final_output)

            # Store chat history
            chat_history = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": final_output}
            ]
            store_chat_history(mobilenumber, chat_history)

    except Exception as e:
        print("An error occurred:", str(e))


if __name__ == "__main__":
    run_crew()