from scraper import ArticleScraper
from rag_pipeline import EmbeddingGenerator, VectorStore, ContextRetriever
from llm_client import GeminiClient, ConversationManager
import concurrent.futures
import hashlib


class ChatService:
    """Orchestrates chatbot components: scraping, RAG, and LLM"""

    def __init__(self):
        self.scraper = ArticleScraper()
        self.embedding_gen = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.context_retriever = ContextRetriever(vector_store=self.vector_store)
        self.llm_client = GeminiClient()
        self.conversation_manager = ConversationManager()

    def process_message(self, ticker, message, frontend_context, conversation_id):
        """
        Process a user message and generate a response

        Args:
            ticker: Stock ticker symbol
            message: User message
            frontend_context: Context from frontend (overview, financials, news, etc.)
            conversation_id: Unique conversation identifier

        Yields:
            Response chunks for streaming
        """
        try:
            # Build comprehensive context
            prompt = self._assemble_prompt(
                query=message,
                ticker=ticker,
                frontend_context=frontend_context,
                conversation_id=conversation_id
            )

            # Get conversation history
            history = self.conversation_manager.get_history(conversation_id)

            # Stream response from LLM
            full_response = ""
            for chunk in self.llm_client.stream_response(prompt, history):
                full_response += chunk
                yield chunk

            # Save to conversation history
            self.conversation_manager.add_message(conversation_id, 'user', message)
            self.conversation_manager.add_message(conversation_id, 'assistant', full_response)

        except Exception as e:
            print(f"Error processing message: {e}")
            yield "I encountered an error processing your request. Please try again."

    def scrape_and_embed_articles(self, ticker, articles):
        """
        Background job to scrape and embed news articles

        Args:
            ticker: Stock ticker symbol
            articles: List of article metadata from Polygon API

        Returns:
            Dictionary with scraping statistics
        """
        results = {
            "scraped": 0,
            "embedded": 0,
            "failed": 0,
            "skipped": 0
        }

        def process_article(article):
            """Process a single article"""
            try:
                # Generate unique document ID
                article_url = article.get('article_url', '')
                doc_id = f"{ticker}_news_{self._hash_url(article_url)}"

                # Check if already processed
                if self.vector_store.document_exists(doc_id):
                    return {'status': 'skipped'}

                # Scrape full content
                content = self.scraper.scrape_article(article_url)

                if not content:
                    # Fall back to article description if scraping fails
                    content = article.get('description', '')
                    if not content or len(content) < 50:
                        return {'status': 'failed'}

                # Generate embedding
                embedding = self.embedding_gen.generate_embedding(content)

                if not embedding:
                    return {'status': 'failed'}

                # Prepare metadata
                metadata = {
                    "ticker": ticker,
                    "type": "news_article",
                    "title": article.get('title', ''),
                    "url": article_url,
                    "published_date": article.get('published_utc', ''),
                    "source": article.get('publisher', {}).get('name', 'Unknown'),
                    "content_preview": content[:200],
                    "full_content": content  # Store full content in metadata
                }

                # Store in FAISS
                success = self.vector_store.upsert_document(doc_id, embedding, metadata)

                if success:
                    return {'status': 'embedded'}
                else:
                    return {'status': 'failed'}

            except Exception as e:
                print(f"Error processing article: {e}")
                return {'status': 'failed'}

        # Process articles in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_article, article) for article in articles[:20]]  # Limit to 20 articles

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                status = result.get('status', 'failed')

                if status == 'embedded':
                    results['embedded'] += 1
                    results['scraped'] += 1
                elif status == 'skipped':
                    results['skipped'] += 1
                elif status == 'failed':
                    results['failed'] += 1

        # Save FAISS index after batch operations
        if results['embedded'] > 0:
            self.vector_store.save()

        return results

    def _assemble_prompt(self, query, ticker, frontend_context, conversation_id):
        """
        Assemble comprehensive prompt with all context

        Args:
            query: User query
            ticker: Stock ticker
            frontend_context: Data from frontend
            conversation_id: Conversation ID

        Returns:
            Complete prompt string
        """
        prompt_parts = []

        # Add current stock overview
        overview = frontend_context.get('overview', {})
        if overview:
            prompt_parts.append(self._format_overview(ticker, overview))

        # Retrieve relevant context from RAG
        rag_contexts = self.context_retriever.retrieve_context(query, ticker)
        if rag_contexts:
            prompt_parts.append(self._format_rag_contexts(rag_contexts))

        # Add financials if relevant to query
        if any(keyword in query.lower() for keyword in ['revenue', 'profit', 'income', 'earnings', 'financial', 'balance']):
            financials = frontend_context.get('financials')
            if financials:
                prompt_parts.append(self._format_financials(financials))

        # Add dividends if relevant
        if 'dividend' in query.lower():
            dividends = frontend_context.get('dividends')
            if dividends:
                prompt_parts.append(self._format_dividends(dividends))

        # Add splits if relevant
        if 'split' in query.lower():
            splits = frontend_context.get('splits')
            if splits:
                prompt_parts.append(self._format_splits(splits))

        # Add sentiment if relevant to query
        sentiment_keywords = ['sentiment', 'bullish', 'bearish', 'feel', 'opinion',
                             'mood', 'social', 'twitter', 'reddit', 'stocktwits', 'buzz']
        if any(keyword in query.lower() for keyword in sentiment_keywords):
            # Retrieve sentiment posts from RAG
            sentiment_contexts = self._retrieve_sentiment_context(query, ticker)
            if sentiment_contexts:
                prompt_parts.append(self._format_sentiment_contexts(sentiment_contexts))

            # Include aggregate sentiment from frontend if available
            sentiment_data = frontend_context.get('sentiment')
            if sentiment_data:
                prompt_parts.append(self._format_aggregate_sentiment(sentiment_data))

        # Combine all context
        context_str = "\n\n---\n\n".join(prompt_parts) if prompt_parts else "No additional context available."

        # Final prompt
        full_prompt = f"""Context Information:
{context_str}

---

User Question: {query}

Please provide a data-driven answer based on the context above."""

        return full_prompt

    def _format_overview(self, ticker, overview):
        """Format stock overview data"""
        details = overview.get('details', {}).get('results', {})
        prev_close = overview.get('previousClose', {}).get('results', [{}])[0]

        company_name = details.get('name', ticker)
        description = details.get('description', 'No description available')[:300]
        market_cap = details.get('market_cap', 0)

        close_price = prev_close.get('c', 0)
        volume = prev_close.get('v', 0)
        high = prev_close.get('h', 0)
        low = prev_close.get('l', 0)

        return f"""Stock Overview for {ticker} - {company_name}:
- Current Price: ${close_price:,.2f}
- Market Cap: ${market_cap:,.0f}
- Volume: {volume:,.0f}
- Day High: ${high:,.2f}
- Day Low: ${low:,.2f}
- Description: {description}"""

    def _format_rag_contexts(self, contexts):
        """Format RAG retrieved contexts"""
        if not contexts:
            return ""

        formatted = ["Relevant News Articles:"]

        for ctx in contexts[:5]:  # Top 5 results
            metadata = ctx['metadata']
            title = metadata.get('title', 'Untitled')
            source = metadata.get('source', 'Unknown')
            date = metadata.get('published_date', '')[:10]  # Just the date
            content = metadata.get('full_content', metadata.get('content_preview', ''))[:500]  # First 500 chars

            formatted.append(f"\n- {title} ({source}, {date})")
            formatted.append(f"  Content: {content}...")

        return "\n".join(formatted)

    def _format_financials(self, financials):
        """Format financial data"""
        results = financials.get('results', [])
        if not results:
            return ""

        formatted = ["Recent Financial Data:"]

        for result in results[:4]:  # Last 4 quarters/years
            fiscal_period = result.get('fiscal_period', '')
            fiscal_year = result.get('fiscal_year', '')

            financials_data = result.get('financials', {})
            income_statement = financials_data.get('income_statement', {})
            balance_sheet = financials_data.get('balance_sheet', {})

            revenue = income_statement.get('revenues', {}).get('value', 0)
            net_income = income_statement.get('net_income_loss', {}).get('value', 0)
            assets = balance_sheet.get('assets', {}).get('value', 0)

            formatted.append(f"\n{fiscal_period} {fiscal_year}:")
            formatted.append(f"  - Revenue: ${revenue:,.0f}")
            formatted.append(f"  - Net Income: ${net_income:,.0f}")
            formatted.append(f"  - Total Assets: ${assets:,.0f}")

        return "\n".join(formatted)

    def _format_dividends(self, dividends):
        """Format dividend data"""
        results = dividends.get('results', [])
        if not results:
            return ""

        formatted = ["Recent Dividends:"]

        for div in results[:5]:
            ex_date = div.get('ex_dividend_date', '')
            amount = div.get('cash_amount', 0)
            formatted.append(f"- {ex_date}: ${amount:.2f} per share")

        return "\n".join(formatted)

    def _format_splits(self, splits):
        """Format stock split data"""
        results = splits.get('results', [])
        if not results:
            return ""

        formatted = ["Stock Splits:"]

        for split in results[:5]:
            execution_date = split.get('execution_date', '')
            split_from = split.get('split_from', 1)
            split_to = split.get('split_to', 1)
            formatted.append(f"- {execution_date}: {split_to}-for-{split_from} split")

        return "\n".join(formatted)

    def _hash_url(self, url):
        """Generate a short hash for URL"""
        return hashlib.md5(url.encode()).hexdigest()[:12]

    def _retrieve_sentiment_context(self, query, ticker):
        """Retrieve sentiment posts from FAISS"""
        try:
            query_embedding = self.embedding_gen.generate_query_embedding(query)
            if not query_embedding:
                return []

            matches = self.vector_store.search(
                query_embedding=query_embedding,
                ticker=ticker,
                namespace="sentiment",
                top_k=5
            )

            contexts = []
            for match in matches:
                contexts.append({
                    'score': match.score,
                    'metadata': match.metadata,
                    'id': match.id
                })

            return contexts
        except Exception as e:
            print(f"Error retrieving sentiment context: {e}")
            return []

    def _format_sentiment_contexts(self, contexts):
        """Format sentiment posts from RAG"""
        if not contexts:
            return ""

        formatted = ["Relevant Social Media Posts:"]

        for ctx in contexts[:5]:
            metadata = ctx['metadata']
            platform = metadata.get('platform', 'unknown')
            sentiment = metadata.get('sentiment_label', 'neutral')
            content = metadata.get('full_content', metadata.get('content', ''))[:300]
            author = metadata.get('author', 'unknown')
            likes = metadata.get('likes', 0)

            formatted.append(f"\n- [{platform.upper()}] @{author} ({sentiment}, {likes} likes)")
            formatted.append(f"  \"{content}...\"")

        return "\n".join(formatted)

    def _format_aggregate_sentiment(self, sentiment_data):
        """Format aggregate sentiment data from frontend"""
        aggregate = sentiment_data.get('aggregate', {})
        if not aggregate:
            return ""

        label = aggregate.get('label', 'neutral')
        score = aggregate.get('score', 0)
        confidence = aggregate.get('confidence', 0)
        post_count = aggregate.get('post_count', 0)
        sources = aggregate.get('sources', {})

        return f"""Current Social Media Sentiment:
- Overall: {label.upper()} (score: {score:.2f}, confidence: {confidence:.0%})
- Posts analyzed: {post_count}
- Sources: StockTwits ({sources.get('stocktwits', 0)}), Reddit ({sources.get('reddit', 0)}), Twitter ({sources.get('twitter', 0)})"""
