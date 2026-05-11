from google.adk.agents import Agent

from prompt import DEEP_LEARNING_AGENT_PROMT
from tools import train_classification_model, train_with_config
from src.config import AGENT_LLM_MODEL

deep_learning_agent = Agent(
    model=AGENT_LLM_MODEL,
    name="deep_learning_agent",
    instruction=DEEP_LEARNING_AGENT_PROMT,
    tools=[train_classification_model, train_with_config],
)
