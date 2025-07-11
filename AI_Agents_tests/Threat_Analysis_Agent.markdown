# Cybersecurity Threat Analysis Agent

In this code sample, you will use the Semantic Kernel AI Framework to create a basic cybersecurity threat analysis agent.

The goal of this sample is to demonstrate how to implement a simple agent for analyzing potential cybersecurity threats using Semantic Kernel.

## Prerequisites
Install the required Python packages:
```bash
pip install semantic-kernel openai python-dotenv
```
Ensure a `.env` file is created with your `GITHUB_TOKEN` for GitHub Models. Example `.env` file:
```plaintext
GITHUB_TOKEN=your_github_token_here
```

## Import the Needed Python Packages
The following Python packages are required for this sample:
```python
import asyncio
import os
from typing import Annotated
from openai import AsyncOpenAI
from dotenv import load_dotenv
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function
import random
```

## Creating the Client
We use GitHub Models to access the Large Language Model (LLM), with the `ai_model_id` set to `gpt-4o-mini`. You can experiment with other models available on the [GitHub Models marketplace](https://github.com/marketplace/models) to observe different results. The `OpenAIChatCompletion` connector is used to interact with the Azure Inference SDK for GitHub Models. Other connectors are available in Semantic Kernel for different model providers.

```python
# Define a sample plugin for cybersecurity threat analysis
class ThreatAnalysisPlugin:
    """A plugin to provide random cybersecurity threat scenarios for analysis."""
    def __init__(self):
        # List of common cybersecurity threats
        self.threats = [
            "Phishing Attack",
            "Ransomware",
            "DDoS Attack",
            "SQL Injection",
            "Man-in-the-Middle Attack",
            "Credential Stuffing",
            "Zero-Day Exploit",
            "Malware Infection",
            "Social Engineering",
            "Insider Threat"
        ]
        # Track last threat to avoid repeats
        self.last_threat = None

    @kernel_function(description="Provides a random cybersecurity threat scenario.")
    def get_random_threat(self) -> Annotated[str, "Returns a random cybersecurity threat scenario."]:
        # Get available threats (excluding last one if possible)
        available_threats = self.threats.copy()
        if self.last_threat and len(available_threats) > 1:
            available_threats.remove(self.last_threat)
        # Select a random threat
        threat = random.choice(available_threats)
        # Update the last threat
        self.last_threat = threat
        return threat

# Load environment variables and validate
load_dotenv()
github_token = os.environ.get("GITHUB_TOKEN")
if not github_token:
    raise ValueError("GITHUB_TOKEN environment variable is not set.")

client = AsyncOpenAI(
    api_key=github_token,
    base_url="https://models.inference.ai.azure.com/",
)

# Create an AI Service that will be used by the ChatCompletionAgent
chat_completion_service = OpenAIChatCompletion(
    ai_model_id="gpt-4o-mini",
    async_client=client,
)
```

## Creating the Agent
Below, we create the `ThreatAnalysisAgent`. For this example, we use simple instructions to guide the agent's behavior. You can modify these instructions to see how the agent responds differently.

```python
agent = ChatCompletionAgent(
    service=chat_completion_service,
    plugins=[ThreatAnalysisPlugin()],
    name="ThreatAnalysisAgent",
    instructions="You are a cybersecurity AI agent that helps analyze potential threats and provides mitigation strategies.",
)
```

## Running the Agent
We run the agent using a `ChatHistoryAgentThread`. Any required system messages are passed to the agent's `invoke_stream` method via the `messages` keyword argument. The user query is set to "Analyze a potential cybersecurity threat." You can modify this input to explore different agent responses.

```python
async def main():
    # Create a new thread for the agent
    thread = None
    user_inputs = ["Analyze a potential cybersecurity threat."]
    for user_input in user_inputs:
        print(f"# User: {user_input}\n")
        first_chunk = True
        async for response in agent.invoke_stream(messages=user_input, thread=thread):
            # Print the response
            if first_chunk:
                print(f"# {response.name}: ", end="", flush=True)
                first_chunk = False
            print(f"{response}", end="", flush=True)
            thread = response.thread
        print()
    # Clean up the thread
    if thread:
        try:
            await thread.delete()
        except Exception as e:
            print(f"Failed to delete thread: {e}")

# Execute the async main function
asyncio.run(main())
```

## Example Output
Running the script with the input "Analyze a potential cybersecurity threat" might produce:
```
# User: Analyze a potential cybersecurity threat.

# ThreatAnalysisAgent: The threat is a Phishing Attack. This involves fraudulent emails attempting to steal sensitive information. Recommended mitigation: Implement email filtering and conduct user awareness training.
```

## Notes
- Ensure your `GITHUB_TOKEN` has access to the `gpt-4o-mini` model on GitHub Models. Check the [GitHub Models marketplace](https://github.com/marketplace/models) for available models.
- The `ThreatAnalysisPlugin` currently provides a random threat. For API-related issues, refer to [xAI's API documentation](https://x.ai/api).