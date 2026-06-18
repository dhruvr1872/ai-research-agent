from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    tavily_api_key: str = Field(default="", env="TAVILY_API_KEY")
    langchain_api_key: str = Field(default="", env="LANGCHAIN_API_KEY")
    langchain_tracing_v2: str = Field(default="false", env="LANGCHAIN_TRACING_V2")
    llm_model: str = Field(default="gpt-4o-mini", env="LLM_MODEL")
    max_search_results: int = Field(default=5, env="MAX_SEARCH_RESULTS")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
