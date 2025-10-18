"""
Educational Context Injection Manager

Manages the building and injection of educational context from enriched metadata
into report agent prompts using Claude Agent SDK PrePrompt hooks.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class ContextInjectionManager:
    """Manages educational context injection for report agents."""
    
    def __init__(self, session_id: str):
        """Initialize context injection manager.
        
        Args:
            session_id: Session identifier
        """
        self.session_id = session_id
        self._context_cache: Optional[str] = None
        self._last_built: Optional[datetime] = None
    
    def get_session_dir(self) -> Path:
        """Get session directory path.
        
        Returns:
            Path to session directory
        """
        return Path.home() / "lrepos" / "claudeagent-multiagent-latest" / "KEVIN" / "sessions" / self.session_id
    
    def build_context(self, force_rebuild: bool = False) -> str:
        """Build educational context from enriched metadata.
        
        Args:
            force_rebuild: Force rebuild even if cached
            
        Returns:
            Formatted educational context string
        """
        # Use cache if available and not forcing rebuild
        if self._context_cache and not force_rebuild:
            if self._last_built:
                age = (datetime.now() - self._last_built).total_seconds()
                if age < 300:  # Cache for 5 minutes
                    logger.debug(f"Using cached context (age: {age:.1f}s)")
                    return self._context_cache
        
        try:
            # Read enriched metadata
            metadata_file = self.get_session_dir() / "enriched_search_metadata.json"
            
            if not metadata_file.exists():
                logger.warning(f"Enriched metadata not found: {metadata_file}")
                return ""
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Filter articles with salient points
            enriched_articles = [
                a for a in metadata.get('search_metadata', [])
                if a.get('has_full_content') and a.get('salient_points')
            ]
            
            if not enriched_articles:
                logger.warning("No articles with salient points found in enriched metadata")
                return ""
            
            # Format context
            formatted = self._format_context(enriched_articles)
            
            # Cache
            self._context_cache = formatted
            self._last_built = datetime.now()
            
            logger.info(f"✅ Built educational context: {len(enriched_articles)} articles, {len(formatted)} chars")
            
            return formatted
            
        except Exception as e:
            logger.error(f"Failed to build educational context: {e}", exc_info=True)
            return ""
    
    def _format_context(self, articles: list) -> str:
        """Format articles into educational context.
        
        Args:
            articles: List of article metadata with salient points
            
        Returns:
            Formatted context string
        """
        context_parts = [
            f"\n{'='*80}",
            f"RESEARCH EDUCATIONAL CONTEXT",
            f"{'='*80}\n",
            f"**{len(articles)} articles with detailed summaries**\n"
        ]
        
        for article in articles:
            context_parts.append(f"""
**Article {article['index']}: {article['title']}**
- Source: {article['source']}
- Date: {article.get('date', 'N/A')}
- Relevance: {article.get('relevance_score', 0):.2f}
- URL: {article['url']}

**Key Points:**
{article['salient_points']}

---
""")
        
        context_parts.append(f"\n{'='*80}\n")
        context_parts.append(f"**Instructions**: Use the specific facts, dates, and figures from these summaries. ")
        context_parts.append(f"Cite sources appropriately using (Source, Date) format. Do not hallucinate information.")
        context_parts.append(f"\n{'='*80}\n\n")
        
        return "\n".join(context_parts)
    
    def create_injection_hook(self):
        """Create a UserPromptSubmit hook for guaranteed context injection.
        
        Returns:
            Async hook function compatible with Claude Agent SDK
        """
        async def user_prompt_submit_hook(input_data: dict, tool_use_id: str | None, context) -> dict:
            """UserPromptSubmit hook that injects educational context.
            
            Args:
                input_data: Hook input containing 'prompt' key
                tool_use_id: Tool use ID (not used for UserPromptSubmit)
                context: Hook context
                
            Returns:
                Hook output with additionalContext
            """
            from .debug_checkpoints import DebugCheckpoint
            
            # Extract prompt from input_data
            prompt = input_data.get('prompt', '')
            
            # Build context
            edu_context = self.build_context()
            
            if not edu_context:
                logger.warning("No educational context available for injection")
                return {}  # No modification
            
            # Save debug checkpoints
            checkpoint = DebugCheckpoint(self.session_id)
            checkpoint.save_text("educational_context", edu_context)
            checkpoint.save_text("original_prompt", prompt)
            checkpoint.save("injection_metadata", {
                "context_length": len(edu_context),
                "prompt_length": len(prompt),
                "injection_successful": True,
                "timestamp": datetime.now().isoformat()
            })
            
            # Log injection
            logger.info("="*80)
            logger.info("EDUCATIONAL CONTEXT INJECTED")
            logger.info(f"Session: {self.session_id}")
            logger.info(f"Context length: {len(edu_context)} chars")
            logger.info(f"Prompt length: {len(prompt)} chars")
            logger.info("="*80)
            
            # Return additionalContext to be injected before the prompt
            return {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": edu_context
                }
            }
        
        return user_prompt_submit_hook
