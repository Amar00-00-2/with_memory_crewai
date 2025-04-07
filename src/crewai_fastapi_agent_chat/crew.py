from crewai import Agent, Crew, Process, Task
from crewai_fastapi_agent_chat.tools.custom_tool import RouterTool
from crewai.memory import LongTermMemory,ShortTermMemory,EntityMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.rag_storage import RAGStorage

from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators
from langchain_openai import ChatOpenAI
import os


# src/latest_ai_development/crew.py
llm_openai = ChatOpenAI(model="gpt-4o", openai_api_key=os.environ['OPENAI_API_KEY'], temperature=0)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@CrewBase
class LatestAiDevelopmentCrew():
  """LatestAiDevelopment crew"""

  @before_kickoff
  def before_kickoff_function(self, inputs):
    print(f"Before kickoff function with inputs: {inputs}")
    return inputs # You can return the inputs or modify them as needed

  @after_kickoff
  def after_kickoff_function(self, result):
    print(f"After kickoff function with result: {result}")
    return result # You can return the result or modify it as needed


@CrewBase
class EmailRagAgent:
    """EmailRagAgent crew"""
    
    agents_config_path = 'config/agents.yaml'
    tasks_config_path = 'config/tasks.yaml'

    def __init__(self, user_id: str):
        self.user_id = user_id
        # self.agents_config = self.load_yaml(self.agents_config_path)
        # self.tasks_config = self.load_yaml(self.tasks_config_path)

    # def load_yaml(self, path):
    #     """Helper method to load YAML files"""
    #     with open(path, 'r') as file:
    #         return yaml.safe_load(file)  # Load the YAML file into a dictionary

    @agent
    def router_agent(self) -> Agent:
        return Agent(
			config=self.agents_config['router_agent'],
			# verbose=True,
			# allow_delegation=True,
			tools=[RouterTool()],
			llm=llm_openai
		)

    @task
    def router_task(self) -> Task:
        return Task(
			config=self.tasks_config['router_task'],
		)
        
    @crew
    def crew(self):
        """Creates the EmailRagAgent crew"""
        # User-specific memory storage paths
        user_ltm_path = f"memory/{self.user_id}/long_term_memory_storage.db"
        user_rag_path = f"memory/{self.user_id}/rag_storage/"

        # Ensure directories exist
        os.makedirs(os.path.dirname(user_ltm_path), exist_ok=True)
        os.makedirs(user_rag_path, exist_ok=True)

        return Crew(
            agents=[self.router_agent()],  
            tasks=[self.router_task()],
            memory=True,
            # Long-term memory for persistent storage
            long_term_memory=LongTermMemory(
                storage=LTMSQLiteStorage(db_path=user_ltm_path)
            ),

            # Short-term memory using RAG
            short_term_memory=ShortTermMemory(
                storage=RAGStorage(
                    embedder_config={
                        "provider": "openai",
                        "config": {
                            "model": 'text-embedding-3-small',
                            "api_key": os.environ["OPENAI_API_KEY"]
                        }
                    },
                    type="short_term",
                    path=user_rag_path
                )
            ),

            # Entity memory for tracking key information
            entity_memory=EntityMemory(
                storage=RAGStorage(
                    embedder_config={
                        "provider": "openai",
                        "config": {
                            "model": 'text-embedding-3-small',
                            "api_key": os.environ["OPENAI_API_KEY"]
                        }
                    },
                    type="entity",
                    path=user_rag_path
                )
            ),
            process=Process.sequential,
            # verbose=True,
        )

