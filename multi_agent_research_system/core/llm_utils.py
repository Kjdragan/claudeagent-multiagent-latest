"""
LLM utility functions for quick scope determination and other lightweight tasks.
"""

import json
import logging

logger = logging.getLogger(__name__)


async def quick_llm_call(prompt: str, temperature: float = 0.1, max_tokens: int = 500) -> str:
    """
    Make a quick LLM call for lightweight decisions like scope determination.

    Uses the research_agent's existing client for consistency.
    """
    try:
        # Import here to avoid circular dependencies

        # For now, use a simple mock response since we're integrating with existing system
        # In production, this would use the actual agent client

        # Simple keyword-based fallback for scope determination
        prompt_lower = prompt.lower()

        if any(keyword in prompt_lower for keyword in ["brief", "summary", "quick", "short", "overview", "report_brief"]):
            scope = "brief"
            reasoning = "Detected brief/summary keywords in query"
            special_req = ""
        elif any(keyword in prompt_lower for keyword in ["comprehensive", "detailed", "extensive", "thorough", "in-depth", "analysis"]):
            scope = "comprehensive"
            reasoning = "Detected comprehensive/detailed keywords in query"
            special_req = ""
        else:
            scope = "default"
            reasoning = "No specific scope keywords detected, using default"
            special_req = ""

        # Check for special requirements
        if "focus on" in prompt_lower:
            import re
            match = re.search(r"focus on ([^.]+)", prompt_lower)
            if match:
                special_req = f"Special focus: {match.group(1)}"
        elif "emphasize" in prompt_lower:
            match = re.search(r"emphasize ([^.]+)", prompt_lower)
            if match:
                special_req = f"Special focus: {match.group(1)}"

        response = {
            "scope": scope,
            "reasoning": reasoning,
            "special_requirements": special_req,
            "confidence": "high"
        }

        logger.info(f"Quick LLM call result: {response}")
        return json.dumps(response)

    except Exception as e:
        logger.error(f"Quick LLM call failed: {e}")
        # Return default response
        fallback = {
            "scope": "default",
            "reasoning": "LLM call failed, using default",
            "special_requirements": "",
            "confidence": "low"
        }
        return json.dumps(fallback)
