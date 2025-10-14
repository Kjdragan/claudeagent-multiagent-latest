#!/usr/bin/env python3
"""
Query Processing System - Advanced Query Analysis and Optimization

This module provides comprehensive query processing capabilities for the agent-based
research system. It analyzes user queries, validates them, optimizes them for research,
and routes them to appropriate research tools with expansion capabilities.

Key Features:
- Advanced query analysis and validation
- Query optimization for better search results
- Orthogonal query generation for comprehensive coverage
- Query expansion and reformulation strategies
- Research strategy recommendation
- Query routing to appropriate tools and parameters
- Quality assessment and improvement suggestions

Query Processing Capabilities:
- Semantic analysis and intent detection
- Query complexity assessment
- Research scope determination
- Source type recommendation
- Parameter optimization for specific tools
- Multi-dimensional query expansion
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    Advanced query processing system for comprehensive research optimization.

    This class provides sophisticated query analysis, validation, optimization,
    and routing capabilities to maximize research effectiveness.
    """

    def __init__(self):
        """Initialize the query processor with default configurations."""

        # Query processing configuration
        self.config = {
            "min_query_length": 3,
            "max_query_length": 500,
            "default_target_results": 50,
            "max_orthogonal_queries": 3,
            "query_expansion_enabled": True,
            "quality_threshold": 0.7
        }

        # Research patterns and keywords
        self.research_patterns = {
            "temporal": ["latest", "recent", "current", "2024", "2025", "today", "now", "new"],
            "comparative": ["vs", "versus", "compare", "comparison", "difference", "better", "worst"],
            "analytical": ["analysis", "analyze", "impact", "effect", "implications", "consequences"],
            "factual": ["what", "when", "where", "who", "how many", "statistics", "data"],
            "technical": ["technical", "specifications", "architecture", "implementation", "details"],
            "academic": ["research", "study", "paper", "academic", "scholarly", "journal"]
        }

        # Stop words for query cleaning
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might", "can"
        }

        logger.info("🔧 Query processor initialized with advanced analysis capabilities")

    async def analyze_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a user query.

        Args:
            query: User's research query
            context: Optional context information (mode, requirements, etc.)

        Returns:
            Dict[str, Any]: Comprehensive query analysis results
        """

        logger.info(f"🔍 Analyzing query: {query[:100]}...")

        try:
            # Basic query validation
            validation_result = self._validate_query(query)
            if not validation_result["is_valid"]:
                return {
                    "original_query": query,
                    "validation": validation_result,
                    "analysis_status": "failed",
                    "error": validation_result["error"]
                }

            # Extract query characteristics
            characteristics = self._extract_characteristics(query)

            # Determine research intent
            intent_analysis = self._analyze_intent(query, characteristics)

            # Assess query complexity
            complexity_assessment = self._assess_complexity(query, characteristics, intent_analysis)

            # Recommend research strategy
            strategy_recommendation = self._recommend_strategy(
                query, characteristics, intent_analysis, complexity_assessment, context
            )

            # Build comprehensive analysis result
            analysis_result = {
                "original_query": query,
                "validation": validation_result,
                "characteristics": characteristics,
                "intent_analysis": intent_analysis,
                "complexity_assessment": complexity_assessment,
                "strategy_recommendation": strategy_recommendation,
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_status": "completed"
            }

            logger.info(f"✅ Query analysis completed: {strategy_recommendation['research_type']}")
            return analysis_result

        except Exception as e:
            logger.error(f"❌ Query analysis failed: {e}")
            return {
                "original_query": query,
                "analysis_status": "error",
                "error": str(e)
            }

    def _validate_query(self, query: str) -> Dict[str, Any]:
        """Validate basic query requirements."""

        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": []
        }

        # Check length
        if len(query.strip()) < self.config["min_query_length"]:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Query too short (minimum {self.config['min_query_length']} characters)")

        if len(query) > self.config["max_query_length"]:
            validation_result["warnings"].append(f"Query very long (>{self.config['max_query_length']} characters)")

        # Check for meaningful content
        meaningful_words = [word for word in query.lower().split() if word not in self.stop_words]
        if len(meaningful_words) < 2:
            validation_result["warnings"].append("Query contains few meaningful keywords")

        # Check for potentially problematic content
        problematic_patterns = [
            r'^\s*test\s*$',  # Just "test"
            r'^\s*hello\s*$',  # Just "hello"
            r'^[\W_]+$',  # Only symbols
        ]

        for pattern in problematic_patterns:
            if re.match(pattern, query.strip(), re.IGNORECASE):
                validation_result["is_valid"] = False
                validation_result["errors"].append("Query appears to be a test or invalid")
                validation_result["error"] = "Query appears to be a test or invalid"
                break

        if not validation_result["is_valid"] and "error" not in validation_result:
            validation_result["error"] = "; ".join(validation_result["errors"])

        return validation_result

    def _extract_characteristics(self, query: str) -> Dict[str, Any]:
        """Extract detailed characteristics from the query."""

        characteristics = {
            "length": len(query),
            "word_count": len(query.split()),
            "keyword_density": {},
            "research_patterns": [],
            "entities": [],
            "temporal_indicators": [],
            "technical_terms": [],
            "question_type": None
        }

        # Analyze word patterns
        words = [word.lower().strip('.,!?;:') for word in query.split()]
        meaningful_words = [word for word in words if word not in self.stop_words]

        # Calculate keyword density
        for word in meaningful_words:
            characteristics["keyword_density"][word] = meaningful_words.count(word) / len(meaningful_words)

        # Detect research patterns
        for pattern_type, keywords in self.research_patterns.items():
            matched_keywords = [kw for kw in keywords if kw in query.lower()]
            if matched_keywords:
                characteristics["research_patterns"].append({
                    "type": pattern_type,
                    "keywords": matched_keywords,
                    "strength": len(matched_keywords) / len(keywords)
                })

        # Extract temporal indicators
        temporal_keywords = self.research_patterns["temporal"]
        characteristics["temporal_indicators"] = [kw for kw in temporal_keywords if kw in query.lower()]

        # Detect question type
        question_patterns = {
            "what": r'^\s*what\s',
            "how": r'^\s*how\s',
            "why": r'^\s*why\s',
            "when": r'^\s*when\s',
            "where": r'^\s*where\s',
            "who": r'^\s*who\s',
            "which": r'^\s*which\s'
        }

        for q_type, pattern in question_patterns.items():
            if re.match(pattern, query.strip(), re.IGNORECASE):
                characteristics["question_type"] = q_type
                break

        return characteristics

    def _analyze_intent(self, query: str, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the user's research intent."""

        intent_analysis = {
            "primary_intent": "general_research",
            "secondary_intents": [],
            "confidence_scores": {},
            "research_goals": [],
            "scope_indicators": {}
        }

        # Determine primary intent based on patterns
        patterns = characteristics.get("research_patterns", [])
        temporal_strength = next((p["strength"] for p in patterns if p["type"] == "temporal"), 0)
        comparative_strength = next((p["strength"] for p in patterns if p["type"] == "comparative"), 0)
        analytical_strength = next((p["strength"] for p in patterns if p["type"] == "analytical"), 0)
        factual_strength = next((p["strength"] for p in patterns if p["type"] == "factual"), 0)
        technical_strength = next((p["strength"] for p in patterns if p["type"] == "technical"), 0)

        # Calculate confidence scores for different intents
        intent_scores = {
            "temporal_research": temporal_strength,
            "comparative_analysis": comparative_strength,
            "deep_analysis": analytical_strength,
            "fact_finding": factual_strength,
            "technical_research": technical_strength
        }

        # Determine primary intent
        primary_intent = max(intent_scores, key=intent_scores.get)
        intent_analysis["primary_intent"] = primary_intent
        intent_analysis["confidence_scores"] = intent_scores

        # Identify secondary intents (scores > 0.2)
        for intent, score in intent_scores.items():
            if intent != primary_intent and score > 0.2:
                intent_analysis["secondary_intents"].append(intent)

        # Determine research goals
        if temporal_strength > 0.3:
            intent_analysis["research_goals"].append("current_information")
        if comparative_strength > 0.3:
            intent_analysis["research_goals"].append("comparison_evaluation")
        if analytical_strength > 0.3:
            intent_analysis["research_goals"].append("deep_understanding")
        if factual_strength > 0.3:
            intent_analysis["research_goals"].append("accurate_information")
        if technical_strength > 0.3:
            intent_analysis["research_goals"].append("technical_details")

        # Assess scope indicators
        query_words = query.lower().split()
        broad_indicators = ["overview", "general", "comprehensive", "all", "everything"]
        specific_indicators = ["specific", "particular", "detailed", "exact", "precise"]

        if any(indicator in query_words for indicator in broad_indicators):
            intent_analysis["scope_indicators"]["breadth"] = "broad"
        elif any(indicator in query_words for indicator in specific_indicators):
            intent_analysis["scope_indicators"]["breadth"] = "specific"
        else:
            intent_analysis["scope_indicators"]["breadth"] = "moderate"

        return intent_analysis

    def _assess_complexity(self, query: str, characteristics: Dict[str, Any],
                          intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the complexity of the research query."""

        complexity_assessment = {
            "overall_complexity": "medium",
            "complexity_score": 0.5,
            "factors": {},
            "recommendations": []
        }

        # Base complexity factors
        factors = {
            "length_complexity": min(characteristics["word_count"] / 15, 1.0),
            "pattern_complexity": len(characteristics["research_patterns"]) / 6,
            "intent_complexity": len(intent_analysis["research_goals"]) / 5,
            "entity_complexity": len(characteristics["entities"]) / 10
        }

        complexity_assessment["factors"] = factors

        # Calculate overall complexity score
        complexity_score = sum(factors.values()) / len(factors)
        complexity_assessment["complexity_score"] = complexity_score

        # Determine complexity level
        if complexity_score < 0.3:
            complexity_assessment["overall_complexity"] = "simple"
            complexity_assessment["recommendations"].append("Standard research approach should be sufficient")
        elif complexity_score < 0.7:
            complexity_assessment["overall_complexity"] = "moderate"
            complexity_assessment["recommendations"].append("Consider multiple search angles for comprehensive coverage")
        else:
            complexity_assessment["overall_complexity"] = "complex"
            complexity_assessment["recommendations"].append("Requires comprehensive multi-angle research approach")
            complexity_assessment["recommendations"].append("Consider breaking into multiple research phases")

        return complexity_assessment

    def _recommend_strategy(self, query: str, characteristics: Dict[str, Any],
                          intent_analysis: Dict[str, Any], complexity_assessment: Dict[str, Any],
                          context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Recommend optimal research strategy based on analysis."""

        strategy = {
            "research_type": "comprehensive",
            "tool_recommendation": "zplayground1_search_scrape_clean",
            "parameter_optimization": {},
            "query_expansion_needed": True,
            "estimated_success_rate": 0.8,
            "processing_time_estimate": "moderate"
        }

        # Base parameter optimization
        base_params = {
            "num_results": self.config["default_target_results"],
            "auto_crawl_top": 20,
            "anti_bot_level": 1,
            "session_prefix": "comprehensive_research"
        }

        # Adjust parameters based on analysis
        complexity = complexity_assessment["overall_complexity"]
        primary_intent = intent_analysis["primary_intent"]

        if complexity == "simple":
            base_params["num_results"] = 30
            base_params["auto_crawl_top"] = 15
            strategy["processing_time_estimate"] = "fast"
            strategy["estimated_success_rate"] = 0.9
        elif complexity == "complex":
            base_params["num_results"] = 70
            base_params["auto_crawl_top"] = 25
            base_params["anti_bot_level"] = 2
            strategy["processing_time_estimate"] = "slow"
            strategy["estimated_success_rate"] = 0.7

        # Intent-specific optimizations
        if primary_intent == "temporal_research":
            base_params["session_prefix"] = "temporal_research"
            strategy["query_expansion_needed"] = True
        elif primary_intent == "comparative_analysis":
            base_params["session_prefix"] = "comparative_analysis"
            strategy["query_expansion_needed"] = True
        elif primary_intent == "technical_research":
            base_params["session_prefix"] = "technical_research"
            base_params["anti_bot_level"] = max(base_params["anti_bot_level"], 2)

        # Context-based adjustments
        if context:
            user_mode = context.get("mode", "web")
            if user_mode == "academic":
                base_params["num_results"] = int(base_params["num_results"] * 1.2)
                base_params["session_prefix"] = "academic_research"
            elif user_mode == "news":
                base_params["session_prefix"] = "news_research"
                strategy["query_expansion_needed"] = True

            user_requirements = context.get("user_requirements", {})
            target_results = user_requirements.get("target_results")
            if target_results:
                base_params["num_results"] = target_results

        strategy["parameter_optimization"] = base_params

        return strategy

    async def optimize_query(self, query: str, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize the original query and generate expanded queries.

        Args:
            query: Original user query
            analysis_result: Results from query analysis

        Returns:
            Dict[str, Any]: Optimized queries with expansion strategies
        """

        logger.info(f"🔧 Optimizing query: {query[:50]}...")

        try:
            optimization_result = {
                "original_query": query,
                "optimized_primary_query": self._optimize_primary_query(query, analysis_result),
                "expanded_queries": [],
                "orthogonal_queries": [],
                "optimization_strategy": "comprehensive_expansion",
                "optimization_timestamp": datetime.now().isoformat()
            }

            # Generate expanded queries if needed
            strategy = analysis_result.get("strategy_recommendation", {})
            if strategy.get("query_expansion_needed", True):
                optimization_result["expanded_queries"] = self._generate_expanded_queries(
                    query, analysis_result
                )
                optimization_result["orthogonal_queries"] = self._generate_orthogonal_queries(
                    query, analysis_result
                )

            logger.info(f"✅ Query optimization completed: {len(optimization_result['expanded_queries'])} expanded queries")
            return optimization_result

        except Exception as e:
            logger.error(f"❌ Query optimization failed: {e}")
            return {
                "original_query": query,
                "optimized_primary_query": query,
                "expanded_queries": [],
                "orthogonal_queries": [],
                "optimization_status": "error",
                "error": str(e)
            }

    def _optimize_primary_query(self, query: str, analysis_result: Dict[str, Any]) -> str:
        """Optimize the primary query for better search results."""

        optimized_query = query.strip()

        # Add temporal context if needed
        intent = analysis_result.get("intent_analysis", {})
        if intent.get("primary_intent") == "temporal_research":
            temporal_indicators = analysis_result.get("characteristics", {}).get("temporal_indicators", [])
            if not temporal_indicators:
                # Add "latest" or "recent" if not present
                optimized_query = f"latest {optimized_query}"

        # Remove unnecessary words
        words = optimized_query.split()
        meaningful_words = [word for word in words if word.lower() not in self.stop_words]
        optimized_query = " ".join(meaningful_words)

        return optimized_query

    def _generate_expanded_queries(self, query: str, analysis_result: Dict[str, Any]) -> List[str]:
        """Generate expanded queries for comprehensive coverage."""

        expanded_queries = []

        characteristics = analysis_result.get("characteristics", {})
        intent = analysis_result.get("intent_analysis", {})

        # Original query variations
        base_query = self._optimize_primary_query(query, analysis_result)

        # Add contextual terms based on intent
        if intent.get("primary_intent") == "temporal_research":
            temporal_variations = [
                f"latest {base_query}",
                f"recent developments in {base_query}",
                f"current {base_query} trends",
                f"{base_query} 2024 2025"
            ]
            expanded_queries.extend(temporal_variations[:2])

        elif intent.get("primary_intent") == "comparative_analysis":
            comparative_variations = [
                f"{base_query} comparison",
                f"{base_query} vs alternatives",
                f"benefits drawbacks {base_query}",
                f"{base_query} evaluation"
            ]
            expanded_queries.extend(comparative_variations[:2])

        elif intent.get("primary_intent") == "technical_research":
            technical_variations = [
                f"technical specifications {base_query}",
                f"{base_query} implementation details",
                f"{base_query} architecture design",
                f"how {base_query} works"
            ]
            expanded_queries.extend(technical_variations[:2])

        else:
            # General research expansions
            general_variations = [
                f"{base_query} overview",
                f"{base_query} guide",
                f"understanding {base_query}",
                f"{base_query} explained"
            ]
            expanded_queries.extend(general_variations[:2])

        return expanded_queries[:3]  # Limit to 3 expanded queries

    def _generate_orthogonal_queries(self, query: str, analysis_result: Dict[str, Any]) -> List[str]:
        """Generate orthogonal queries for different research angles."""

        orthogonal_queries = []

        characteristics = analysis_result.get("characteristics", {})
        base_query = self._optimize_primary_query(query, analysis_result)

        # Generate orthogonal approaches based on research goals
        intent = analysis_result.get("intent_analysis", {})
        research_goals = intent.get("research_goals", [])

        # Different orthogonal strategies
        if "current_information" in research_goals:
            orthogonal_queries.append(f"breaking news {base_query}")

        if "comparison_evaluation" in research_goals:
            orthogonal_queries.append(f"{base_query} market analysis")

        if "deep_understanding" in research_goals:
            orthogonal_queries.append(f"scientific research {base_query}")

        if "technical_details" in research_goals:
            orthogonal_queries.append(f"{base_query} expert analysis")

        # Add default orthogonal queries if none generated
        if not orthogonal_queries:
            orthogonal_queries = [
                f"{base_query} case studies",
                f"{base_query} industry reports",
                f"{base_query} expert opinions"
            ]

        return orthogonal_queries[:self.config["max_orthogonal_queries"]]

    async def route_query(self, optimized_queries: Dict[str, Any],
                         strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route optimized queries to appropriate research tools and parameters.

        Args:
            optimized_queries: Result from query optimization
            strategy: Research strategy recommendation

        Returns:
            Dict[str, Any]: Query routing configuration
        """

        logger.info("🚀 Routing queries to research tools")

        try:
            routing_config = {
                "primary_tool": strategy.get("tool_recommendation", "zplayground1_search_scrape_clean"),
                "tool_parameters": strategy.get("parameter_optimization", {}),
                "query_sequence": [],
                "execution_plan": "parallel_expansion",
                "routing_timestamp": datetime.now().isoformat()
            }

            # Build query execution sequence
            primary_query = optimized_queries.get("optimized_primary_query")
            expanded_queries = optimized_queries.get("expanded_queries", [])
            orthogonal_queries = optimized_queries.get("orthogonal_queries", [])

            # Primary query
            routing_config["query_sequence"].append({
                "query": primary_query,
                "priority": "primary",
                "parameters": routing_config["tool_parameters"]
            })

            # Expanded queries
            for i, exp_query in enumerate(expanded_queries):
                exp_params = routing_config["tool_parameters"].copy()
                exp_params["num_results"] = max(10, exp_params["num_results"] // 3)  # Reduce results for expansions

                routing_config["query_sequence"].append({
                    "query": exp_query,
                    "priority": "secondary",
                    "sequence": i + 1,
                    "parameters": exp_params
                })

            # Orthogonal queries (if any)
            for i, orth_query in enumerate(orthogonal_queries):
                orth_params = routing_config["tool_parameters"].copy()
                orth_params["num_results"] = max(5, orth_params["num_results"] // 5)  # Further reduce for orthogonal

                routing_config["query_sequence"].append({
                    "query": orth_query,
                    "priority": "orthogonal",
                    "sequence": i + 1,
                    "parameters": orth_params
                })

            logger.info(f"✅ Query routing completed: {len(routing_config['query_sequence'])} queries configured")
            return routing_config

        except Exception as e:
            logger.error(f"❌ Query routing failed: {e}")
            return {
                "primary_tool": "zplayground1_search_scrape_clean",
                "tool_parameters": {"num_results": 50},
                "query_sequence": [{"query": optimized_queries.get("optimized_primary_query", ""), "priority": "primary"}],
                "routing_status": "error",
                "error": str(e)
            }

    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        End-to-end query processing pipeline.

        Args:
            query: User's research query
            context: Optional context information

        Returns:
            Dict[str, Any]: Complete query processing results
        """

        logger.info(f"🔄 Starting end-to-end query processing: {query[:50]}...")

        try:
            # Step 1: Analyze query
            analysis_result = await self.analyze_query(query, context)
            if analysis_result.get("analysis_status") != "completed":
                return {
                    "original_query": query,
                    "processing_status": "failed",
                    "stage": "analysis",
                    "error": analysis_result.get("error", "Unknown analysis error")
                }

            # Step 2: Optimize query
            optimization_result = await self.optimize_query(query, analysis_result)

            # Step 3: Route queries
            strategy = analysis_result.get("strategy_recommendation", {})
            routing_result = await self.route_query(optimization_result, strategy)

            # Build complete processing result
            processing_result = {
                "original_query": query,
                "context": context,
                "analysis": analysis_result,
                "optimization": optimization_result,
                "routing": routing_result,
                "processing_timestamp": datetime.now().isoformat(),
                "processing_status": "completed",
                "ready_for_execution": True
            }

            logger.info(f"✅ Query processing completed successfully")
            return processing_result

        except Exception as e:
            logger.error(f"❌ End-to-end query processing failed: {e}")
            return {
                "original_query": query,
                "processing_status": "error",
                "error": str(e),
                "ready_for_execution": False
            }


# Fallback query processor for when advanced features aren't available
class FallbackQueryProcessor:
    """Simplified query processor for fallback operations."""

    def __init__(self):
        self.config = {"default_target_results": 50}

    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Basic query processing fallback."""

        return {
            "original_query": query,
            "context": context,
            "analysis": {
                "validation": {"is_valid": True, "warnings": [], "errors": []},
                "strategy_recommendation": {
                    "research_type": "comprehensive",
                    "tool_recommendation": "zplayground1_search_scrape_clean",
                    "parameter_optimization": {
                        "num_results": self.config["default_target_results"],
                        "auto_crawl_top": 20,
                        "anti_bot_level": 1
                    }
                }
            },
            "optimization": {
                "optimized_primary_query": query.strip(),
                "expanded_queries": [],
                "orthogonal_queries": []
            },
            "routing": {
                "primary_tool": "zplayground1_search_scrape_clean",
                "query_sequence": [{"query": query.strip(), "priority": "primary"}]
            },
            "processing_status": "completed_fallback",
            "ready_for_execution": True
        }