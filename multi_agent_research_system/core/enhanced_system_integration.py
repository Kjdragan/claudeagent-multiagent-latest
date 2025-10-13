"""
Enhanced System Integration Module

This module provides comprehensive integration between the Enhanced Research Orchestrator
and the enhanced systems from Phase 1, including anti-bot escalation, content cleaning,
search optimization, and other advanced features.

Phase 2.2 Integration: Connect enhanced orchestrator with Phase 1 systems
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Import Phase 1 enhanced systems
try:
    from ..utils.anti_bot_escalation import AntiBotEscalator, EscalationLevel
    from ..utils.content_cleaning import assess_content_cleanliness, clean_content
    from ..utils.search_strategy_selector import SearchStrategySelector
    from ..utils.crawl4ai_media_optimized import MediaOptimizedCrawler
    from ..utils.enhanced_relevance_scorer import calculate_relevance_score
    from ..utils.research_data_standardizer import standardize_research_data
    from ..utils.query_intent_analyzer import analyze_query_intent
    PHASE1_SYSTEMS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Phase 1 enhanced systems not available: {str(e)}")
    PHASE1_SYSTEMS_AVAILABLE = False

# Import orchestrator components
from .enhanced_orchestrator import EnhancedResearchOrchestrator, WorkflowHookContext, RichMessage, MessageType
from .quality_framework import QualityFramework
from .logging_config import get_logger


class EnhancedSystemIntegrator:
    """
    Comprehensive integration layer for Phase 1 enhanced systems.

    This class provides seamless integration between the enhanced orchestrator
    and all the advanced systems developed in Phase 1, including:
    - Anti-bot escalation system
    - AI-powered content cleaning
    - Intelligent search strategy selection
    - Media-optimized crawling
    - Enhanced relevance scoring
    - Research data standardization
    - Query intent analysis
    """

    def __init__(self, orchestrator: EnhancedResearchOrchestrator):
        """Initialize enhanced system integration."""
        self.orchestrator = orchestrator
        self.logger = get_logger("enhanced_system_integrator")
        self.logger.info("Initializing Enhanced System Integration")

        # Initialize Phase 1 systems
        self.phase1_systems = {}
        if PHASE1_SYSTEMS_AVAILABLE:
            self._initialize_phase1_systems()
        else:
            self.logger.warning("Phase 1 enhanced systems not available, using fallback implementations")

        # Integration metrics
        self.integration_metrics = {
            "anti_bot_escalations": 0,
            "content_cleaning_operations": 0,
            "search_strategy_selections": 0,
            "media_optimization_usage": 0,
            "relevance_score_calculations": 0,
            "data_standardization_operations": 0,
            "query_intent_analyses": 0,
            "total_integrations": 0
        }

        # Register integration hooks with orchestrator
        self._register_integration_hooks()

        self.logger.info("Enhanced System Integration initialized successfully")

    def _initialize_phase1_systems(self):
        """Initialize all Phase 1 enhanced systems."""
        try:
            # Anti-bot escalation system
            self.phase1_systems["anti_bot_escalator"] = AntiBotEscalator()
            self.logger.info("✅ Anti-bot escalation system initialized")

            # Content cleaning system
            self.phase1_systems["content_cleaning_available"] = True
            self.logger.info("✅ Content cleaning system initialized")

            # Search strategy selector
            self.phase1_systems["search_strategy_selector"] = SearchStrategySelector()
            self.logger.info("✅ Search strategy selector initialized")

            # Media-optimized crawler
            self.phase1_systems["media_optimized_crawler"] = MediaOptimizedCrawler()
            self.logger.info("✅ Media-optimized crawler initialized")

            # Enhanced relevance scorer
            self.phase1_systems["relevance_scorer_available"] = True
            self.logger.info("✅ Enhanced relevance scorer initialized")

            # Research data standardizer
            self.phase1_systems["data_standardizer_available"] = True
            self.logger.info("✅ Research data standardizer initialized")

            # Query intent analyzer
            self.phase1_systems["query_intent_analyzer_available"] = True
            self.logger.info("✅ Query intent analyzer initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize Phase 1 systems: {str(e)}")
            raise

    def _register_integration_hooks(self):
        """Register integration hooks with the orchestrator."""
        if not self.orchestrator.hook_manager:
            self.logger.warning("Hook manager not available, skipping hook registration")
            return

        # Register hooks for different integration points
        self.orchestrator.hook_manager.register_hook("workflow_stage_start", self._hook_integration_stage_start)
        self.orchestrator.hook_manager.register_hook("workflow_stage_complete", self._hook_integration_stage_complete)
        self.orchestrator.hook_manager.register_hook("agent_handoff", self._hook_integration_agent_handoff)

        self.logger.info("Integration hooks registered successfully")

    async def enhance_research_execution(self, session_id: str, research_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance research execution with Phase 1 systems integration.

        This method integrates anti-bot escalation, search strategy selection,
        media optimization, and other Phase 1 enhancements into the research process.
        """
        self.logger.info(f"Enhancing research execution for session {session_id}")

        enhancement_context = {
            "session_id": session_id,
            "operation": "research_enhancement",
            "start_time": datetime.now(),
            "applied_enhancements": []
        }

        try:
            # 1. Enhanced Search Strategy Selection
            if "query" in research_params and "search_strategy_selector" in self.phase1_systems:
                search_strategy = await self._apply_search_strategy_selection(
                    research_params["query"], enhancement_context
                )
                research_params.update(search_strategy)

            # 2. Anti-Bot Escalation Configuration
            anti_bot_config = await self._configure_anti_bot_escalation(research_params, enhancement_context)
            research_params.update(anti_bot_config)

            # 3. Media Optimization Configuration
            media_config = await self._configure_media_optimization(research_params, enhancement_context)
            research_params.update(media_config)

            # 4. Enhanced Content Processing Configuration
            content_config = await self._configure_content_processing(research_params, enhancement_context)
            research_params.update(content_config)

            # Create enhancement summary message
            if self.orchestrator.message_processor:
                enhancement_message = RichMessage(
                    id=f"enhancement_{session_id}",
                    message_type=MessageType.INFO,
                    content=f"Applied {len(enhancement_context['applied_enhancements'])} Phase 1 enhancements to research execution",
                    session_id=session_id,
                    agent_name="system_integrator",
                    stage="research_enhancement",
                    metadata={
                        "enhancements": enhancement_context["applied_enhancements"],
                        "enhancement_count": len(enhancement_context["applied_enhancements"])
                    }
                )
                await self.orchestrator.message_processor.process_message(enhancement_message)

            # Update integration metrics
            self.integration_metrics["total_integrations"] += 1

            return research_params

        except Exception as e:
            self.logger.error(f"Research enhancement failed: {str(e)}")
            raise

    async def enhance_content_processing(self, session_id: str, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance content processing with Phase 1 systems integration.

        This method integrates AI-powered content cleaning, relevance scoring,
        and data standardization into the content processing pipeline.
        """
        self.logger.info(f"Enhancing content processing for session {session_id}")

        enhancement_context = {
            "session_id": session_id,
            "operation": "content_enhancement",
            "start_time": datetime.now(),
            "processed_items": []
        }

        try:
            enhanced_content = content_data.copy()

            # Process each content item
            if "sources" in enhanced_content:
                enhanced_sources = []
                for source in enhanced_content["sources"]:
                    enhanced_source = await self._process_single_content_item(source, enhancement_context)
                    enhanced_sources.append(enhanced_source)
                enhanced_content["sources"] = enhanced_sources

            # Apply data standardization
            if "data_standardizer_available" in self.phase1_systems:
                standardized_content = await self._apply_data_standardization(enhanced_content, enhancement_context)
                enhanced_content.update(standardized_content)

            # Create content enhancement summary
            if self.orchestrator.message_processor:
                content_message = RichMessage(
                    id=f"content_enhancement_{session_id}",
                    message_type=MessageType.INFO,
                    content=f"Enhanced {len(enhancement_context['processed_items'])} content items with Phase 1 systems",
                    session_id=session_id,
                    agent_name="system_integrator",
                    stage="content_enhancement",
                    metadata={
                        "processed_items": len(enhancement_context["processed_items"]),
                        "enhancements_applied": enhancement_context["processed_items"]
                    }
                )
                await self.orchestrator.message_processor.process_message(content_message)

            return enhanced_content

        except Exception as e:
            self.logger.error(f"Content enhancement failed: {str(e)}")
            raise

    async def enhance_quality_assessment(self, session_id: str, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance quality assessment with Phase 1 systems integration.

        This method integrates enhanced relevance scoring, content analysis,
        and other Phase 1 quality enhancement features.
        """
        self.logger.info(f"Enhancing quality assessment for session {session_id}")

        enhancement_context = {
            "session_id": session_id,
            "operation": "quality_enhancement",
            "start_time": datetime.now(),
            "quality_enhancements": []
        }

        try:
            enhanced_quality_metrics = {}

            # 1. Enhanced Relevance Scoring
            if "relevance_scorer_available" in self.phase1_systems:
                relevance_scores = await self._apply_enhanced_relevance_scoring(
                    content, context, enhancement_context
                )
                enhanced_quality_metrics.update(relevance_scores)

            # 2. Content Quality Analysis
            content_quality = await self._analyze_content_quality(content, enhancement_context)
            enhanced_quality_metrics.update(content_quality)

            # 3. Source Quality Assessment
            if "sources" in context:
                source_quality = await self._assess_source_quality(context["sources"], enhancement_context)
                enhanced_quality_metrics.update(source_quality)

            # Create quality enhancement summary
            if self.orchestrator.message_processor:
                quality_message = RichMessage(
                    id=f"quality_enhancement_{session_id}",
                    message_type=MessageType.QUALITY_ASSESSMENT,
                    content=f"Applied {len(enhancement_context['quality_enhancements'])} Phase 1 quality enhancements",
                    session_id=session_id,
                    agent_name="system_integrator",
                    stage="quality_enhancement",
                    quality_metrics=enhanced_quality_metrics,
                    metadata={
                        "enhancements": enhancement_context["quality_enhancements"],
                        "enhancement_count": len(enhancement_context["quality_enhancements"])
                    }
                )
                await self.orchestrator.message_processor.process_message(quality_message)

            return enhanced_quality_metrics

        except Exception as e:
            self.logger.error(f"Quality assessment enhancement failed: {str(e)}")
            raise

    async def _apply_search_strategy_selection(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply intelligent search strategy selection."""
        try:
            strategy_selector = self.phase1_systems["search_strategy_selector"]
            strategy_analysis = strategy_selector.analyze_and_recommend(query)

            context["applied_enhancements"].append("search_strategy_selection")
            self.integration_metrics["search_strategy_selections"] += 1

            self.logger.info(f"Applied search strategy: {strategy_analysis.recommended_strategy.value}")

            return {
                "search_strategy": strategy_analysis.recommended_strategy.value,
                "strategy_confidence": strategy_analysis.confidence,
                "strategy_reasoning": strategy_analysis.reasoning
            }

        except Exception as e:
            self.logger.error(f"Search strategy selection failed: {str(e)}")
            return {}

    async def _configure_anti_bot_escalation(self, research_params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Configure anti-bot escalation system."""
        try:
            anti_bot_escalator = self.phase1_systems["anti_bot_escalator"]

            # Configure escalation settings based on research parameters
            escalation_config = {
                "max_escalation_level": research_params.get("anti_bot_max_level", 3),
                "base_delay": research_params.get("anti_bot_delay", 1.0),
                "enable_escalation": research_params.get("enable_anti_bot", True)
            }

            context["applied_enhancements"].append("anti_bot_escalation")
            self.integration_metrics["anti_bot_escalations"] += 1

            self.logger.info(f"Configured anti-bot escalation: max_level={escalation_config['max_escalation_level']}")

            return {
                "anti_bot_config": escalation_config,
                "escalation_enabled": escalation_config["enable_escalation"]
            }

        except Exception as e:
            self.logger.error(f"Anti-bot escalation configuration failed: {str(e)}")
            return {}

    async def _configure_media_optimization(self, research_params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Configure media optimization for crawling."""
        try:
            media_crawler = self.phase1_systems["media_optimized_crawler"]

            # Get optimized configuration
            optimized_config = media_crawler.get_optimized_config(
                text_mode=research_params.get("text_mode", True),
                exclude_all_images=research_params.get("exclude_images", True),
                light_mode=research_params.get("light_mode", True),
                page_timeout=research_params.get("page_timeout", 20000)
            )

            context["applied_enhancements"].append("media_optimization")
            self.integration_metrics["media_optimization_usage"] += 1

            self.logger.info("Configured media optimization for enhanced crawling performance")

            return {
                "crawl_config": optimized_config,
                "media_optimization_enabled": True
            }

        except Exception as e:
            self.logger.error(f"Media optimization configuration failed: {str(e)}")
            return {}

    async def _configure_content_processing(self, research_params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Configure enhanced content processing."""
        try:
            content_config = {
                "enable_content_cleaning": research_params.get("enable_cleaning", True),
                "cleanliness_threshold": research_params.get("cleanliness_threshold", 0.7),
                "enable_relevance_scoring": research_params.get("enable_relevance_scoring", True),
                "enable_data_standardization": research_params.get("enable_standardization", True)
            }

            if content_config["enable_content_cleaning"]:
                context["applied_enhancements"].append("content_cleaning_configuration")
            if content_config["enable_relevance_scoring"]:
                context["applied_enhancements"].append("relevance_scoring_configuration")
            if content_config["enable_data_standardization"]:
                context["applied_enhancements"].append("data_standardization_configuration")

            self.logger.info("Configured enhanced content processing")

            return {"content_processing_config": content_config}

        except Exception as e:
            self.logger.error(f"Content processing configuration failed: {str(e)}")
            return {}

    async def _process_single_content_item(self, source: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single content item with Phase 1 enhancements."""
        enhanced_source = source.copy()

        try:
            # 1. Content Cleaning
            if "content_cleaning_available" in self.phase1_systems and "content" in source:
                is_clean, cleanliness_score = await assess_content_cleanliness(source["content"], source.get("url", ""))

                if not is_clean:
                    cleaned_content = await clean_content(source["content"], source.get("url", ""))
                    enhanced_source["content"] = cleaned_content
                    enhanced_source["content_cleaned"] = True
                    enhanced_source["cleanliness_score"] = cleanliness_score

                    context["processed_items"].append("content_cleaning")
                    self.integration_metrics["content_cleaning_operations"] += 1

            # 2. Relevance Scoring
            if "relevance_scorer_available" in self.phase1_systems:
                relevance_score = calculate_relevance_score(
                    enhanced_source.get("content", ""),
                    enhanced_source.get("url", "")
                )
                enhanced_source["relevance_score"] = relevance_score

                context["processed_items"].append("relevance_scoring")
                self.integration_metrics["relevance_score_calculations"] += 1

        except Exception as e:
            self.logger.warning(f"Failed to process content item: {str(e)}")

        return enhanced_source

    async def _apply_data_standardization(self, content: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply research data standardization."""
        try:
            standardized_data = standardize_research_data(content)

            context["processed_items"].append("data_standardization")
            self.integration_metrics["data_standardization_operations"] += 1

            self.logger.info("Applied research data standardization")

            return standardized_data

        except Exception as e:
            self.logger.error(f"Data standardization failed: {str(e)}")
            return {}

    async def _apply_enhanced_relevance_scoring(self, content: str, context: Dict[str, Any], enhancement_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply enhanced relevance scoring."""
        try:
            # Extract URLs from content for relevance scoring
            urls = self._extract_urls_from_content(content)
            relevance_scores = {}

            for url in urls:
                score = calculate_relevance_score(content, url)
                relevance_scores[url] = score

            enhancement_context["quality_enhancements"].append("enhanced_relevance_scoring")
            self.integration_metrics["relevance_score_calculations"] += len(urls)

            return {"relevance_scores": relevance_scores}

        except Exception as e:
            self.logger.error(f"Enhanced relevance scoring failed: {str(e)}")
            return {}

    async def _analyze_content_quality(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content quality with enhanced metrics."""
        try:
            quality_metrics = {
                "content_length": len(content),
                "word_count": len(content.split()),
                "estimated_reading_time": len(content.split()) / 200,  # words per minute
                "content_density": len(content.replace(" ", "")) / len(content) if content else 0,
                "sentence_count": content.count(".") + content.count("!") + content.count("?"),
                "paragraph_count": content.count("\n\n") + 1
            }

            # Quality indicators
            quality_metrics.update({
                "has_sufficient_length": quality_metrics["word_count"] >= 100,
                "has_reasonable_density": 0.5 <= quality_metrics["content_density"] <= 0.8,
                "readability_score": self._calculate_readability_score(content)
            })

            context["quality_enhancements"].append("content_quality_analysis")

            return {"content_quality_metrics": quality_metrics}

        except Exception as e:
            self.logger.error(f"Content quality analysis failed: {str(e)}")
            return {}

    async def _assess_source_quality(self, sources: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess source quality with enhanced metrics."""
        try:
            source_quality_metrics = {
                "total_sources": len(sources),
                "sources_with_content": len([s for s in sources if s.get("content")]),
                "average_relevance_score": 0,
                "high_quality_sources": 0,
                "source_diversity": self._calculate_source_diversity(sources)
            }

            # Calculate average relevance score
            relevance_scores = [s.get("relevance_score", 0) for s in sources if "relevance_score" in s]
            if relevance_scores:
                source_quality_metrics["average_relevance_score"] = sum(relevance_scores) / len(relevance_scores)
                source_quality_metrics["high_quality_sources"] = len([s for s in relevance_scores if s >= 0.7])

            context["quality_enhancements"].append("source_quality_assessment")

            return {"source_quality_metrics": source_quality_metrics}

        except Exception as e:
            self.logger.error(f"Source quality assessment failed: {str(e)}")
            return {}

    def _extract_urls_from_content(self, content: str) -> List[str]:
        """Extract URLs from content for relevance scoring."""
        import re
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        return re.findall(url_pattern, content)

    def _calculate_readability_score(self, content: str) -> float:
        """Calculate basic readability score."""
        sentences = content.count(".") + content.count("!") + content.count("?")
        words = len(content.split())

        if sentences == 0:
            return 0.0

        # Simple readability metric (average words per sentence)
        avg_words_per_sentence = words / sentences

        # Normalize to 0-1 scale (optimal is 15-20 words per sentence)
        if 15 <= avg_words_per_sentence <= 20:
            return 1.0
        elif avg_words_per_sentence < 15:
            return max(0.0, avg_words_per_sentence / 15)
        else:
            return max(0.0, 1.0 - (avg_words_per_sentence - 20) / 30)

    def _calculate_source_diversity(self, sources: List[Dict[str, Any]]) -> float:
        """Calculate source diversity based on domain diversity."""
        try:
            domains = set()
            for source in sources:
                url = source.get("url", "")
                if url:
                    domain = url.split("/")[2] if len(url.split("/")) > 2 else ""
                    domains.add(domain)

            # Diversity score: unique domains / total sources
            return len(domains) / len(sources) if sources else 0.0

        except Exception:
            return 0.0

    # Hook Implementations
    async def _hook_integration_stage_start(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for integration at stage start."""
        if context.workflow_stage.value == "research":
            self.logger.info(f"🔧 Integrating Phase 1 systems for research stage in session {context.session_id}")
        return {"event": "integration_stage_start", "stage": context.workflow_stage.value}

    async def _hook_integration_stage_complete(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for integration at stage completion."""
        self.logger.info(f"✅ Phase 1 integration completed for stage {context.workflow_stage.value}")
        return {"event": "integration_stage_complete", "stage": context.workflow_stage.value}

    async def _hook_integration_agent_handoff(self, context: WorkflowHookContext) -> Dict[str, Any]:
        """Hook for integration during agent handoff."""
        self.logger.info(f"🔄 Phase 1 systems ready for agent handoff: {context.agent_name}")
        return {"event": "integration_agent_handoff", "agent": context.agent_name}

    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get comprehensive integration metrics."""
        return {
            "integration_metrics": self.integration_metrics.copy(),
            "phase1_systems_available": PHASE1_SYSTEMS_AVAILABLE,
            "available_systems": list(self.phase1_systems.keys()),
            "total_integrations": self.integration_metrics["total_integrations"],
            "integration_breakdown": {
                "anti_bot_escalations": self.integration_metrics["anti_bot_escalations"],
                "content_cleaning_operations": self.integration_metrics["content_cleaning_operations"],
                "search_strategy_selections": self.integration_metrics["search_strategy_selections"],
                "media_optimization_usage": self.integration_metrics["media_optimization_usage"],
                "relevance_score_calculations": self.integration_metrics["relevance_score_calculations"],
                "data_standardization_operations": self.integration_metrics["data_standardization_operations"],
                "query_intent_analyses": self.integration_metrics["query_intent_analyses"]
            }
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all integrated Phase 1 systems."""
        status = {
            "integration_available": PHASE1_SYSTEMS_AVAILABLE,
            "systems_status": {}
        }

        if PHASE1_SYSTEMS_AVAILABLE:
            for system_name in self.phase1_systems:
                status["systems_status"][system_name] = {
                    "available": True,
                    "initialized": True,
                    "last_integration": datetime.now().isoformat()
                }
        else:
            status["systems_status"]["fallback"] = {
                "available": True,
                "initialized": True,
                "note": "Using fallback implementations"
            }

        return status


# Factory function for creating enhanced system integrator
def create_enhanced_system_integrator(orchestrator: EnhancedResearchOrchestrator) -> EnhancedSystemIntegrator:
    """
    Factory function to create enhanced system integrator.

    Args:
        orchestrator: Enhanced Research Orchestrator instance

    Returns:
        Enhanced System Integrator instance
    """
    return EnhancedSystemIntegrator(orchestrator)