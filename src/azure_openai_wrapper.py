from openai import AzureOpenAI
import logging
import os

logger = logging.getLogger(__name__)

class AzureOpenAIWrapper:
    """
    Azure OpenAI API wrapper class designed to be compatible with LangChain's interfaces.
    API key, version, and endpoint can be configured through environment variables or parameters.
    """

    # Default values - can be overridden by environment variables or constructor params
    DEFAULT_API_KEY = "c9be74243d9e4db78346018b44592e9a"
    DEFAULT_API_VERSION = "2024-02-15-preview"
    DEFAULT_AZURE_ENDPOINT = "https://formaigpt.openai.azure.com"
    DEFAULT_MODEL = "gpt-35-turbo"  

    def __init__(self, 
                model=None, 
                temperature=0,
                api_key=None,
                api_version=None,
                azure_endpoint=None):
        """
        Initialize the Azure OpenAI wrapper with the specified model and settings.
        
        Args:
            model: Azure OpenAI model deployment name
            temperature: Temperature parameter for text generation
            api_key: Azure OpenAI API key (will override environment variable)
            api_version: Azure OpenAI API version
            azure_endpoint: Azure OpenAI endpoint URL
        """
        # Get values from environment variables or use provided parameters
        self.model = model or os.getenv("AZURE_OPENAI_DEPLOYMENT", self.DEFAULT_MODEL)
        self.temperature = temperature
        
        # Log which model deployment we're using
        logger.info(f"Using Azure OpenAI deployment: {self.model}")
        
        # Initialize Azure OpenAI client
        try:
            self.__client = AzureOpenAI(
                api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY", self.DEFAULT_API_KEY),
                api_version=api_version or os.getenv("AZURE_OPENAI_API_VERSION", self.DEFAULT_API_VERSION),
                azure_endpoint=azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", self.DEFAULT_AZURE_ENDPOINT),
            )
            logger.debug(f"AzureOpenAIWrapper initialized with model {self.model}")
        except Exception as e:
            logger.error(f"Error initializing AzureOpenAIWrapper: {str(e)}")
            raise

    def invoke(self, messages):
        """
        Invoke the Azure OpenAI API with the given messages.
        
        Compatible with LangChain's interface:
        - Accepts string, list of dictionaries, list of tuples, or LangChain message objects
        - Returns string content directly for easy integration
        
        Args:
            messages: Input in various formats (string, list, etc.)
            
        Returns:
            String containing the model's response
        """
        # Process input messages to extract the prompt
        prompt = self._process_messages(messages)
        
        try:
            # Call Azure OpenAI API
            response = self.__client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
                temperature=self.temperature,
                max_tokens=3000,
            )
            
            # Extract content from response
            content = response.choices[0].message.content
            
            return content
        except Exception as e:
            error_msg = f"Error in AzureOpenAIWrapper.invoke: {str(e)}"
            logger.error(error_msg)
            return f"Error: {str(e)}"

    def _process_messages(self, messages):
        """
        Process various message formats into a single prompt string.
        
        Args:
            messages: Input in various formats
            
        Returns:
            Processed prompt as a string
        """
        # Handle string input
        if isinstance(messages, str):
            return messages
        
        # Handle list input (dicts, tuples, or LangChain messages)
        elif isinstance(messages, list):
            prompt = ""
            for msg in messages:
                # Dictionary format {"role": "user", "content": "..."}
                if isinstance(msg, dict):
                    if msg.get("role") in ["user", "human"]:
                        prompt += msg.get("content", "") + "\n"
                
                # Tuple format ("user", "content")
                elif isinstance(msg, tuple) and len(msg) == 2:
                    role, content = msg
                    if role in ["user", "human"]:
                        prompt += content + "\n"
                
                # LangChain message objects
                elif hasattr(msg, 'type') and hasattr(msg, 'content'):
                    if msg.type == 'human':
                        prompt += msg.content + "\n"
            return prompt
        
        # Fallback for other types
        else:
            return str(messages)

    def __call__(self, messages):
        """Make the wrapper callable like a function."""
        return self.invoke(messages)