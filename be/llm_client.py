from openai import OpenAI
import tiktoken
from datetime import datetime, timedelta
from config import OPENAI_API_KEY, CHAT_MODEL, MAX_CONTEXT_LENGTH


class OpenAIClient:
    """Handles interactions with OpenAI GPT-4 API"""

    SYSTEM_PROMPT = """You are an expert stock market analyst assistant integrated into a trading platform.
You have access to:
- Real-time stock data (price, volume, market cap)
- Historical price charts and trends
- Financial statements (income statement, balance sheet)
- News articles with full content
- Dividend and split history
- Related companies

When answering questions:
- Be concise and data-driven
- Cite specific numbers from the data provided
- Reference news articles when relevant (mention article titles and sources)
- Highlight important trends or anomalies
- Never provide financial advice, only analysis
- If asked about predictions, explain uncertainty and limitations
- Format numbers with proper units (e.g., $1.5B, 10.5M shares)

Always ground your responses in the provided data. If information is not available, say so clearly."""

    def __init__(self, api_key=None, model=None):
        self.client = OpenAI(api_key=api_key or OPENAI_API_KEY)
        self.model = model or CHAT_MODEL
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def generate_response(self, prompt, conversation_history=None):
        """
        Generate a response from GPT-4

        Args:
            prompt: User prompt with context
            conversation_history: List of previous messages

        Returns:
            Generated response text
        """
        try:
            messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

            # Add conversation history
            if conversation_history:
                messages.extend(conversation_history)

            # Add current prompt
            messages.append({"role": "user", "content": prompt})

            # Trim if too long
            messages = self._trim_messages(messages)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error generating response: {e}")
            return f"I encountered an error processing your request. Please try again."

    def stream_response(self, prompt, conversation_history=None):
        """
        Generate a streaming response from GPT-4

        Args:
            prompt: User prompt with context
            conversation_history: List of previous messages

        Yields:
            Response chunks as they arrive
        """
        try:
            messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

            # Add conversation history
            if conversation_history:
                messages.extend(conversation_history)

            # Add current prompt
            messages.append({"role": "user", "content": prompt})

            # Trim if too long
            messages = self._trim_messages(messages)

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            print(f"Error streaming response: {e}")
            yield f"I encountered an error processing your request. Please try again."

    def count_tokens(self, text):
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    def _trim_messages(self, messages):
        """Trim messages to fit within context window"""
        # Calculate total tokens
        total_tokens = sum(self.count_tokens(str(msg['content'])) for msg in messages)

        # If within limit, return as-is
        if total_tokens <= MAX_CONTEXT_LENGTH:
            return messages

        # Keep system message and trim conversation history
        trimmed = [messages[0]]  # System prompt

        # Add messages from most recent, working backwards
        for msg in reversed(messages[1:]):
            msg_tokens = self.count_tokens(str(msg['content']))
            if total_tokens - msg_tokens > MAX_CONTEXT_LENGTH:
                total_tokens -= msg_tokens
            else:
                trimmed.insert(1, msg)

        return trimmed


class ConversationManager:
    """Manages conversation history for chat sessions"""

    def __init__(self):
        self.conversations = {}
        self.ttl_hours = 24

    def add_message(self, conversation_id, role, content):
        """
        Add a message to conversation history

        Args:
            conversation_id: Unique conversation identifier
            role: Message role (user/assistant)
            content: Message content
        """
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                'messages': [],
                'created_at': datetime.now()
            }

        self.conversations[conversation_id]['messages'].append({
            'role': role,
            'content': content
        })

        # Clean up old conversations
        self._cleanup_old_conversations()

    def get_history(self, conversation_id, last_n=5):
        """
        Get conversation history

        Args:
            conversation_id: Unique conversation identifier
            last_n: Number of recent exchanges to return

        Returns:
            List of recent messages
        """
        if conversation_id not in self.conversations:
            return []

        messages = self.conversations[conversation_id]['messages']

        # Return last N exchanges (user + assistant pairs)
        return messages[-(last_n * 2):] if len(messages) > last_n * 2 else messages

    def clear_conversation(self, conversation_id):
        """Clear a conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]

    def _cleanup_old_conversations(self):
        """Remove conversations older than TTL"""
        cutoff = datetime.now() - timedelta(hours=self.ttl_hours)

        to_delete = [
            conv_id for conv_id, data in self.conversations.items()
            if data['created_at'] < cutoff
        ]

        for conv_id in to_delete:
            del self.conversations[conv_id]
