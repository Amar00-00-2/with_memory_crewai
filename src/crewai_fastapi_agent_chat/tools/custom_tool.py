from crewai.tools import BaseTool
from typing import Type,Optional,List
from pydantic import BaseModel, Field,ValidationError
import logging
from dotenv import load_dotenv
load_dotenv()
import os
from openai import AzureOpenAI
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.prompts import SystemMessagePromptTemplate,HumanMessagePromptTemplate,ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage,AIMessage
# from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_pinecone import PineconeVectorStore
import pymysql
from pinecone import Pinecone
from langchain_cerebras import ChatCerebras

from langchain_openai import AzureOpenAIEmbeddings
import re
import json
from crewai_fastapi_agent_chat.notifications.mail_trigger import send_email_template,send_python_email
from crewai_fastapi_agent_chat.notifications.whatsapp_trigger import whati
# Must precede any llm module imports

# from langtrace_python_sdk import langtrace

# langtrace.init(api_key = 'e212c0c85f5af68e1bf5567dea41dd3ecb2779da3789cf651393e4347ca45154')
# Load environment variables
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = "@#CP@202&*(#R"
MYSQL_DB = os.getenv("MYSQL_DB")
DB_PORT=3306

# MONGO_URL = os.getenv("MONGO_URL")
# MONGO_STAGING_DB_NAME = os.getenv("MONGO_STAGING_DB_NAME")
# MONGO_VECTOR_COLLECTION = os.getenv("MONGO_VECTOR_COLLECTION")
# MONGO_DB_VECTOR_INDEX_OPENAI = os.getenv("MONGO_DB_VECTOR_INDEX_OPENAI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL")
OPENAI_EMBEDDING_MODEL = os.environ["OPENAI_EMBEDDING_MODEL"]
AZURE_OPEANI_EMBEDDING_END_POINT = os.environ["AZURE_OPEANI_EMBEDDING_END_POINT"]
AZURE_OPEANI_API_KEY = os.environ["AZURE_OPEANI_API_KEY"]
AZURE_OPEANI_EMBEDDING_VERSION = os.environ["AZURE_OPEANI_EMBEDDING_VERSION"]
CHAT_MODEL = os.getenv("CHAT_MODEL")
AZURE_OPENAI_CHAT_API_KEY = os.environ['AZURE_OPENAI_CHAT_API_KEY']
AZURE_OPENAI_CHAT_END_POINT = os.environ['AZURE_OPENAI_CHAT_END_POINT']
AZURE_OPENAI_CHAT_VERSION = os.environ['AZURE_OPENAI_CHAT_VERSION']

STAGING_PINECONE_INDEX_NAME=os.environ['STAGING_PINECONE_INDEX_NAME']
STAGING_PINECONE_API_KEY= os.getenv("STAGING_PINECONE_API_KEY")
# pinecine setup
LIVE_PINECONE_API_KEY= os.getenv("LIVE_PINECONE_API_KEY")
LIVE_PINECONE_INDEX_NAME=os.environ['LIVE_PINECONE_INDEX_NAME']

pc = Pinecone(api_key=STAGING_PINECONE_API_KEY)
pine_index = pc.Index(STAGING_PINECONE_INDEX_NAME)

# Connect to MySQL
mysql_connection = pymysql.connect(
    host=MYSQL_HOST,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database=MYSQL_DB,
    port=DB_PORT
)

suvi=mysql_connection.cursor


# MongoDB setup
# client = MongoClient(MONGO_URL)
# vector_collection = client[MONGO_STAGING_DB_NAME][MONGO_VECTOR_COLLECTION]
# product_collection = client[MONGO_STAGING_DB_NAME]['products']
# doctype_collection = client[MONGO_STAGING_DB_NAME]['doctypes']
# productfile_collection = client[MONGO_STAGING_DB_NAME]['productfiles']
# files_collection = client[MONGO_STAGING_DB_NAME]['files']

# Embeddings and LLM setup
embeddings=AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPEANI_EMBEDDING_END_POINT,
    api_key=AZURE_OPEANI_API_KEY,
    azure_deployment=OPENAI_EMBEDDING_MODEL,
    openai_api_version=AZURE_OPEANI_EMBEDDING_VERSION
)
llm = ChatCerebras(model=os.environ['CEREBRAS_MODEL'],api_key=os.environ['CEREBRAS_KEY'],temperature=0)
llm_openai = ChatOpenAI(model="gpt-4o", openai_api_key=os.environ['OPENAI_API_KEY'], temperature=0)

# Initialize Azure OpenAI client
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_CHAT_END_POINT, 
    api_key=AZURE_OPENAI_CHAT_API_KEY,  
    api_version=AZURE_OPENAI_CHAT_VERSION,
    azure_deployment=CHAT_MODEL
)

def find_doctype(question,product_doctype):
  print("question>>>>>>>>", ", ".join(map(str, question)) if isinstance(question, list) else str(question))
  function_data = [
      {
          "name": "json_format",
          "description": "Extract the document type from the user's query.",
          "parameters": {
              "type": "object",
              "properties": {
                  "doctype": {
                      "type": "string",
                      "enum": product_doctype + ["None"],  # Restrict output
                      "description": f"Must be one of: {', '.join(product_doctype)}, or 'None' if no match."
                  }
              },
              "required": ["doctype"],
          }
      }
  ]
  # Call Azure OpenAI
  response = openai_client.chat.completions.create(
      model=CHAT_MODEL,
      temperature=0,
      messages=[
          {"role": "system", "content": "Extract the document type from the user's query."},
          {"role": "user", "content": ", ".join(map(str, question)) if isinstance(question, list) else str(question)}
      ],
      functions=function_data,
      function_call={"name": "json_format"}  # Forcing function call
  )
  # Handling the response
  try:
      function_call_data = response.choices[0].message.function_call
      if function_call_data:
          ai_response = json.loads(function_call_data.arguments)
          print("AI Response:", ai_response)
          return ai_response
      else:
          print("No function call executed.")
  except Exception as e:
      print("Error processing response:", e)

def doc_file_url(chat_history,question, productid):
    try:
        # Step 1: Get distinct docType values
        with mysql_connection.cursor() as cursor:
            query = "SELECT DISTINCT docType FROM web_chat_bot_doc_types"
            cursor.execute(query)
            product_doctype = [item[0] for item in cursor.fetchall()]  # Flatten result to list
        print("Available document types:", product_doctype)


        doctype_data=find_doctype(question,product_doctype)
        if doctype_data['doctype'] =="None":
            last_messages = [msg["message"] for msg in chat_history[-5:]]
            doctype_data = find_doctype(last_messages,product_doctype) 
        doctype_value = doctype_data.get('doctype', '').upper()
        # Validate extracted document type
        valid_doctype = next((doctype for doctype in product_doctype if doctype.upper() == doctype_value), None)
        print("Validated doctype:", valid_doctype)

        if valid_doctype is None:
            print("Email is not goin to triggered...")
            return {
                "status": False,
                "message": "Document type is missing or invalid. Please provide a valid document type, or specify if you would like to chat with an agent."
            }

        # Step 3: Query MySQL for the matched document type ID
        with mysql_connection.cursor() as cursor:
            query = """
                SELECT id FROM web_chat_bot_doc_types 
                WHERE docType = %s
                LIMIT 1
            """
            cursor.execute(query, (valid_doctype,))
            doc = cursor.fetchone()

            if not doc:
                return {"status": False, "message": "The specified document type was not found in the database."}

            doc_type_id = doc[0]
            print("Document Type ID:", doc_type_id)

        # Step 4: Query MySQL for fileS3path using docTypeId and productId
        with mysql_connection.cursor() as cursor:
            query = """
                SELECT fileS3path 
                FROM web_chat_bot_product_files 
                WHERE docTypeId = %s AND productId = %s
                LIMIT 1
            """
            cursor.execute(query, (doc_type_id, productid[0]))
            data = cursor.fetchone()
        print("dataaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",data)
        if data and data[0]:
            file_s3_path = data[0]
            print("File S3 Path:", file_s3_path)  # Ensure correct retrieval
            return {"status": True, "message": file_s3_path}
        else:
            print("Data is empty or None:", data)
            return {
                "status": False,
                "message": "No file found for the specified document type and product ID. Please check your input or contact support."
            }

    except Exception as e:
        print("Error occurred:", str(e))
        return {"status": False, "message": "An error occurred while processing your request. Please try again later or contact support."}

def get_and_append_chat_history(chat_history):
    print("get_and_append_chat_history",chat_history)

    if chat_history:
        prompt_chat_history = []
        for i in chat_history:
            if i["messageFrom"] == "agent":
                prompt_chat_history.append(HumanMessage(content=i["message"]))
            if i["messageFrom"] == "whatsapp":
                prompt_chat_history.append(AIMessage(content=i["message"]))
    else:
        prompt_chat_history = []
    return prompt_chat_history
   
def doc_chat_rag_tool(question: str, collection_id: str,chat_history:list,prompt:str,vector_db:str):
    """This tool is used to chat with the document.""" 
    if vector_db=="staging":
        print("staging calledddddddddddddddddddddddddddddddddddd")
        PINECONE_API_KEY = STAGING_PINECONE_API_KEY
        PINECONE_INDEX_NAME = STAGING_PINECONE_INDEX_NAME
    elif vector_db =="live":
        print("live calledddddddddddddddddddddddddddddddddddd")
        PINECONE_API_KEY = LIVE_PINECONE_INDEX_NAME
        PINECONE_INDEX_NAME = LIVE_PINECONE_API_KEY

    pc = Pinecone(api_key=PINECONE_API_KEY)
    pine_index = pc.Index(PINECONE_INDEX_NAME)
    vectorStore = PineconeVectorStore(index=pine_index, embedding=embeddings)

    general_system_template = prompt
    general_user_template = "Question:{question}"
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),

        HumanMessagePromptTemplate.from_template(general_user_template)
    ]
    qa_prompt = ChatPromptTemplate.from_messages( messages )

    retriever = vectorStore.as_retriever(
        query=question,
        filter={"collectionId": collection_id,"isFileDeleted":False},
        search_kwargs={"k": 5},
    )
  
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        combine_docs_chain_kwargs={'prompt': qa_prompt},
        return_source_documents=True,
        retriever=retriever
    )

    with get_openai_callback() as cb:
        chat_history = get_and_append_chat_history(chat_history)
        response = conversation_chain.invoke({"question": question, "chat_history": chat_history})

        result = {
            "answer": response["answer"],
            "usage": {
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "usd_amount": cb.total_cost
            }
        }
    print("Answerrrrr",result["answer"])
    return result['answer']

def email_tool(question: str, email: str,username:str,file_url:str):
    """
    This tool is used to send an email notification.
    
    Args:
    question (str): The input question or command.
    email (str): The default email address to use if no email is provided in the question.
    
    Returns:
    str: Status of the email sending process.
    """
    try:
        print("email_toooollllllllllllll",question,email,username)
        # Convert the question to lowercase for consistency
        question_lower = question.lower().strip()
        print("email_toooollllllllllllll",question_lower)
        # file_url = "https://arulmbucket.s3.ap-south-1.amazonaws.com/DMS_NTE_Brochure.pdf"

        # Case 1: If the question is "email", directly process the default email
        if question_lower == "email":

            print("mail send triggered...")
            extracted_user_name1 = email.split('@')[0]
            send_python_email({'to': email, 'user_name': username, 'file_url': file_url})
            return "Email sent successfully to the default email."

        # Case 2: If the question contains an email, process the email(s) found in the question
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, question)
        
        if emails:
            file_url1=file_url

            print("Actual email : ",emails,file_url1)
            for email_id in emails:
                extracted_user_name = email_id.split('@')[0]
                send_python_email({'to': email_id, 'user_name': username, 'file_url': file_url1})
            return "Email(s) sent successfully to the extracted email address(es)."

        # Case 3: If no email is provided in the question, validate the default email or prompt the user
        else:
            # file_url2 = "https://arulmbucket.s3.ap-south-1.amazonaws.com/DMS_NTE_Brochure.pdf"

            doctype="default"

            file_url2=file_url

            print("Default email",email)
            if not email or '@' not in email:
                return "Please provide a valid email address."
            
            # Use the default email
            extracted_user_name = email.split('@')[0]
            
            send_python_email({'to': email, 'user_name': username, 'file_url': file_url2})
            return "Email sent successfully to the default email."

    except Exception as e:
        return f"Error while sending email: {str(e)}"

def whatsapp_tool(question: str, mobilenumber: str, username: str, file_url: str):
    """This tool is used to send a WhatsApp notification."""
    try:
        # Check if the 'whatsApp' keyword is in the question
        if question.lower() == "whatsapp":
            print("Sending WhatsApp message to:", mobilenumber)
            whati({
                "mobile_number": mobilenumber,
                "user_name": username,
                "file_url": file_url,
                "clientname": username
            })
            return "WhatsApp message sent successfully."

        # Pattern to extract phone numbers
        phone_pattern = r'\+?\d{1,4}?[\s.-]?\(?\d{1,4}?\)?[\s.-]?\d{1,4}[\s.-]?\d{1,4}[\s.-]?\d{1,9}'
        phone_numbers = re.findall(phone_pattern, question)
        
        if phone_numbers:
            file_url1=file_url

            print("Extracted phone numbers:", phone_numbers)
            for phone_number in phone_numbers:
                whati({
                    "mobile_number": mobilenumber,
                    "user_name": username,
                    "file_url": file_url1,
                    "clientname": username
                })
        else:
            doctype="default"

            file_url2=file_url
            # Default to the provided mobile number if no phone numbers are found
            print("No phone numbers found. Using the provided mobile number:", mobilenumber)
            whati({
                "mobile_number": mobilenumber,
                "user_name": username,
                "file_url": file_url2,
                "clientname": username
            })

        return "WhatsApp message sent successfully."
        
    except Exception as e:
        return f"Error while sending WhatsApp message: {str(e)}"

class ChatMessage(BaseModel):
    """Schema for individual chat messages in chat history."""
    messageFrom: str = Field(description="Sender of the message, e.g., 'agent' or 'whatsapp'")
    message: str = Field(description="Content of the message")
    messageType: Optional[str] = Field(default="text", description="Type of message, e.g., 'text', 'image', etc.")
    
class RouterToolInput(BaseModel):
    """Input schema for RouterTool."""
    question: str = Field(description="Question")
    username: str= Field(description="Username")
    collection_id: str= Field(description="Collection id")
    mobilenumber: str = Field(description="Mobile number")
    email: str= Field(description="Email id")
    chat_history: List[ChatMessage] = Field(
        default=[],
        description="List of chat history messages"
    ),
    prompt: str
    vector_db: str
    product_id: List[str] = Field(description="List of product IDs associated with the request")
    product_name: List[str] = Field(description="List of product names associated with the request")
    
class RouterTool(BaseTool):
    name: str = "Router tool"
    description: str = (
        "Analyze asked question and find out the intent of it"
    )
    args_schema: Type[BaseModel] = RouterToolInput

    def _run(self, question: str, username: str, collection_id: str, mobilenumber: str, email: str,chat_history:list,prompt:str,vector_db:str,product_id:list,product_name:list):
        #     Based on the question provided below, determine the following:
        # 1. Is the question asking to send the document via email, or does it only mention the word "email"?
        # 2. Is the question asking to send the document via WhatsApp, or does it only mention the word "WhatsApp"?
        # 3. Is the question asking to send the document, but without specifying whether to use email or WhatsApp?
        # 4. Is the question unrelated to sending any documents?
        openai_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_CHAT_END_POINT,
            api_key=AZURE_OPENAI_CHAT_API_KEY,
            api_version=AZURE_OPENAI_CHAT_VERSION,
            azure_deployment="gpt-4o",
        )

        # Function definition for structured response
        function_data = {
            "name": "document_sending",
            "description": "Handles document sending by requesting recipient details if missing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {"type": "integer", "description": "Intent of action (1-4)."},
                    "recipient_name": {"type": ["string", "null"], "description": "Recipient name."},
                    "email": {"type": ["string", "null"], "description": "Recipient email."},
                    "response": {"type": "string", "description": "System response."},
                    "needs_name": {"type": "boolean", "description": "Needs recipient name?"},
                    "needs_email": {"type": "boolean", "description": "Needs recipient email?"},
                },
                "required": ["intent", "recipient_name", "email", "response", "needs_name", "needs_email"],
            },
        }

        # System instructions
        system_prompt = """
            You are a helpful conversational assistant that helps users send documents. Your tasks are:

            1. Determine the document sharing intent from user messages using these categories:
            - Intent 1: Send document via email
            - Intent 2: Send document via WhatsApp
            - Intent 3: Send document (unspecified method)
            - Intent 4: No intent to send documents

            2. When email sharing is requested (Intent 1), extract any email address mentioned in the conversation.

            3. If an email address is needed but not provided, politely ask the user for it.

            4. Note: WhatsApp number is already available in the system, so no need to ask for it.

            For each user message, return your analysis in this JSON format:
            {
            "intent": [1-4],
            "recipient_name": "[extracted name or null]",
            "email": "[extracted email or null]",
            "response": "[your conversational response to the user]",
            "needs_name": [true/false],
            "needs_email": [true/false]
            }

            Then engage with the user naturally based on your analysis.

            User message: {Question}
        """
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            max_tokens=500,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            functions=[function_data],
            function_call={'name': 'document_sending'}
        )
        import json
        # Parse AI Response
        try:
            json_string = response.choices[0].message.function_call.arguments
            ai_response = json.loads(json_string)
            print("ai_responseee",ai_response)
            if ai_response["intent"] == 1: # Email
                if ai_response['email']:
                    print("Answer: Email triggered")
                    file_url=doc_file_url(chat_history,question,product_id)
                    print("Returned dataaaaaaaaaaa",file_url)
                    if (file_url['status']):
                        print("Email triggered...")
                        return email_tool(question,email, username, file_url['message'])
                    else:
                        return file_url['message']
                else:
                    print("email not triggered..",ai_response['response'])
                    return ai_response['response']
            elif ai_response["intent"] == 2: #Whatsapp
                file_url=doc_file_url(chat_history,question,product_id)
                if (file_url['status']):
                    print("Whatsapp triggered....")
                    return whatsapp_tool(question , mobilenumber, username, file_url['message'])
                else:
                    return file_url['message']
            elif ai_response["intent"] == 3: #Ask
                if ai_response["email"] is None:
                    print("Ask triggered...")
                    return ai_response['response']
            elif ai_response["intent"] == 4: #Rag
                    return doc_chat_rag_tool(question, collection_id,chat_history,prompt,vector_db)
        except ValidationError as e:
            print("❌ Error in RouterResponse conversion:", e)
            return None  # Handle error gracefully

        except Exception as e:
            print("❌ Unexpected error:", e)
            return None

            
  