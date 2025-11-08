from langchain.agents import create_agent
from tools.context_retriver import context_retriever
from config.model_config import SYSTEM_PROMPTS, MODEL, PROVIDER

def get_llm_model(selected_provider="OLLAMA"):
    provider_config = PROVIDERS.get(selected_provider)
    
    if not provider_config:
        raise ValueError(f"Unsupported provider: {selected_provider}")

    if provider_config["provider"] == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=provider_config["MODEL"], temperature=0.4, repeat_penalty=1.3)

    elif provider_config["provider"] == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=provider_config["MODEL"], temperature=0.4, repeat_penalty=1.3)

    else:
        raise ValueError(f"Unsupported provider: {selected_provider}")

def load_agent_executor():
    llm_model = get_llm_model()

    agent_executor = create_agent(model=llm_model, 
                        tools=[context_retriever], 
                        checkpointer=MemorySaver(),
                        system_prompt = SYSTEM_PROMPTS["AGENT_EXECUTOR_PROMPT"]
                        )

    return agent_executor