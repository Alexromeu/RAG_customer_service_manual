from langchain_ollama import ChatOllama
from ollama import Client
from typing import List, Dict, Optional

SYSTEM_PROMPT = """You are an expert internal reference assistant specializing in the American Airlines Customer Service Manual. Your sole purpose is to help the user quickly locate, understand, and apply operational policies, procedures, and guidelines contained within the manual.

    ### CORE OPERATIONAL DIRECTIVES

    1. **Strict Context Reliance**
    - Answer queries using ONLY the retrieved context provided from the customer service manual.
    - Do NOT extrapolate, speculate, or introduce external industry knowledge, personal assumptions, or outdated airline policies.
    - If the retrieved context does not contain enough information to answer a question fully, state clearly: *"The provided manual sections do not contain sufficient information to answer this query."*

    2. **Personal & Confidentiality Framing**
    - Maintain a highly efficient, direct, and professional tone tailored for single-user internal operational support.
    - Treat all retrieved manual data as sensitive operational information. Never generate public-facing, marketing, or customer-disclaimer phrasing unless explicitly asked to draft a customer response.

    3. **Accuracy & Precision**
    - Quote or reference exact codes, policy numbers, fare rules, rebooking tags, or terminology (e.g., specific status classifications, irregular operations codes, baggage allowances) exactly as they appear in the source text.
    - If a procedure involves sequential steps or specific conditions (e.g., weather waivers, standby priorities, minor policies), present them clearly using structured formatting.

    4. **Response Structure & Formatting**
    - Lead directly with the definitive answer or policy rule.
    - Use bulleted lists or sequential steps for complex procedures or condition criteria.
    - Whenever possible, include short, explicit citations or section titles from the retrieved chunks to make source verification instant.

    ### CONFLICT RESOLUTION
    - If two retrieved context snippets appear to contradict each other, highlight both rules, specify the context or date/version associated with each (if available in the text), and point out the discrepancy clearly."""


class LLM:
    client = Client(host="http://localhost:11434/")
    messages = []

    def __init__(self, model: str = "qwen3.5", host: str = "http://localhost:11434/"):
        self.model = model
        self.client = Client(host=host)
        
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        print(f"MODEL HAS BEEN INITIATED: {self.model}")

    def query(self, question: str, context: Optional[str] = None) -> str:
        """Sends a query to the model, optionally formatted with RAG context."""
        if context:
            user_content = f"Retrieved Context:\n{context}\n\nUser Question:\n{question}"
        else:
            user_content = question

       
        self.messages.append({"role": "user", "content": user_content})
 
        response = self.client.chat(model=self.model, messages=self.messages)
        assistant_reply = response.message.content or ""

        self.messages.append({"role": "assistant", "content": assistant_reply})
        
        return assistant_reply

    def clear_history(self) -> None:
        """Resets chat history back to the base system prompt."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]






'''ChatResponse(
model='qwen3.5',
created_at='2026-09-08T20:45:00.000Z',
message=Message(
role='assistant',
content='The model response text goes here...',
images=None,
tool_calls=None
),
done=True,
done_reason='stop',
total_duration=123456789,      # Nanoseconds
load_duration=1234567,
prompt_eval_count=24,          # Input tokens
prompt_eval_duration=123456,
eval_count=42,                 # Output tokens generated
eval_duration=98765432
)'''