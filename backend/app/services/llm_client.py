import os
import logging
from typing import Optional, Any, Dict
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

logger = logging.getLogger("agentic_bridge.services.llm_client")

class LLMProviderFactory:
    """
    Enterprise LLM Provider Factory supporting multi-model orchestration 
    (OpenAI, Anthropic) with automatic credential validation, retry policies, and telemetry.
    """
    
    @staticmethod
    def get_llm(
        provider: Optional[str] = None,
        temperature: float = 0.1,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Any:
        """
        Initializes and returns a LangChain chat model instance based on the specified provider.
        
        Args:
            provider (Optional[str]): The LLM provider ("openai" or "anthropic"). Defaults to environment or "openai".
            temperature (float): Model temperature for generation randomness.
            model_name (Optional[str]): Specific model override (defaults to provider standard).
            api_key (Optional[str]): Optional explicit API key override.
            
        Returns:
            Any: Configured LangChain chat model instance.
        """
        resolved_provider = (provider or os.getenv("DEFAULT_LLM_PROVIDER", "openai")).lower()
        
        if resolved_provider == "openai":
            resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")
            resolved_model = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
            
            if not resolved_key:
                logger.warning("OPENAI_API_KEY is missing from environment variables. LLM execution may fail.")
                
            logger.info(f"Initializing OpenAI Chat model: {resolved_model} (temp: {temperature})")
            return ChatOpenAI(
                model=resolved_model,
                temperature=temperature,
                api_key=resolved_key,
                max_retries=3
            )
            
        elif resolved_provider == "anthropic":
            resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
            resolved_model = model_name or os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-5-sonnet-20241022")
            
            if not resolved_key:
                logger.warning("ANTHROPIC_API_KEY is missing from environment variables. LLM execution may fail.")
                
            logger.info(f"Initializing Anthropic Chat model: {resolved_model} (temp: {temperature})")
            return ChatAnthropic(
                model=resolved_model,
                temperature=temperature,
                api_key=resolved_key,
                max_retries=3
            )
            
        else:
            error_msg = f"Unsupported LLM provider requested: '{resolved_provider}'. Supported providers: 'openai', 'anthropic'."
            logger.error(error_msg)
            raise ValueError(error_msg)

    @staticmethod
    def get_client_metadata() -> Dict[str, Any]:
        """Returns current environment telemetry and active LLM configuration status."""
        return {
            "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
            "default_provider": os.getenv("DEFAULT_LLM_PROVIDER", "openai")
        }