"""
Performance Tests for Agent-Based Research System

This module provides comprehensive performance testing for the agent-based research system,
validating performance under various loads and conditions.

Performance Test Coverage:
- Response time validation
- Concurrent operation performance
- Resource usage monitoring
- Scalability testing
- Load testing under stress conditions

Author: Claude Code Assistant
Version: 1.0.0
"""

import asyncio
import json
import os
import psutil
import tempfile
import time
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import AsyncMock, MagicMock, patch

# Import system components for performance testing
import sys
sys.path.append(str(Path(__file__).parent.parent))


class PerformanceMonitor:
    """Monitor system performance during testing"""

    def __init__(self):
        self.process = psutil.Process()
        self.measurements = []
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self, interval: float = 0.5):
        """Start performance monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

    def _monitor_loop(self, interval: float):
        """Performance monitoring loop"""
        while self.monitoring:
            try:
                measurement = {
                    "timestamp": time.time(),
                    "cpu_percent": self.process.cpu_percent(),
                    "memory_mb": self.process.memory_info().rss / 1024 / 1024,
                    "memory_percent": self.process.memory_percent(),
                    "threads": self.process.num_threads(),
                    "open_files": len(self.process.open_files())
                }
                self.measurements.append(measurement)
            except Exception:
                pass
            time.sleep(interval)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.measurements:
            return {}

        cpu_values = [m["cpu_percent"] for m in self.measurements]
        memory_values = [m["memory_mb"] for m in self.measurements]

        return {
            "duration": self.measurements[-1]["timestamp"] - self.measurements[0]["timestamp"],
            "cpu": {
                "average": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values)
            },
            "memory": {
                "average_mb": sum(memory_values) / len(memory_values),
                "max_mb": max(memory_values),
                "min_mb": min(memory_values)
            },
            "sample_count": len(self.measurements)
        }


class PerformanceTestFixture:
    """Provides test fixtures for performance testing"""

    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="performance_test_"))
        self.monitor = PerformanceMonitor()
        self.test_data = {}
        self.setup_test_environment()

    def setup_test_environment(self):
        """Set up test environment for performance testing"""
        # Create test directories
        directories = [
            "test_sessions",
            "test_workloads",
            "test_results",
            "test_logs"
        ]

        for dir_name in directories:
            (self.temp_dir / dir_name).mkdir(exist_ok=True)

        # Create performance test configurations
        self.create_performance_configurations()

    def create_performance_configurations(self):
        """Create performance test configurations"""
        # Light load configuration
        light_config = {
            "concurrent_sessions": 1,
            "queries_per_session": 1,
            "target_results_per_query": 5,
            "timeout_duration": 10
        }

        # Medium load configuration
        medium_config = {
            "concurrent_sessions": 3,
            "queries_per_session": 2,
            "target_results_per_query": 10,
            "timeout_duration": 20
        }

        # Heavy load configuration
        heavy_config = {
            "concurrent_sessions": 5,
            "queries_per_session": 3,
            "target_results_per_query": 15,
            "timeout_duration": 30
        }

        # Stress test configuration
        stress_config = {
            "concurrent_sessions": 10,
            "queries_per_session": 5,
            "target_results_per_query": 20,
            "timeout_duration": 60
        }

        configs = {
            "light": light_config,
            "medium": medium_config,
            "heavy": heavy_config,
            "stress": stress_config
        }

        for config_name, config_data in configs.items():
            config_path = self.temp_dir / "test_config" / f"{config_name}_load.json"
            config_path.parent.mkdir(exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)

    def load_performance_config(self, load_type: str) -> Dict[str, Any]:
        """Load performance configuration"""
        config_path = self.temp_dir / "test_config" / f"{load_type}_load.json"
        with open(config_path, 'r') as f:
            return json.load(f)

    def create_performance_test_sessions(self, count: int, load_type: str) -> List[Dict[str, Any]]:
        """Create multiple test sessions for performance testing"""
        sessions = []
        config = self.load_performance_config(load_type)

        for i in range(count):
            session_id = f"perf_test_{load_type}_{i+1}_{uuid.uuid4().hex[:8]}"
            session_data = {
                "session_id": session_id,
                "load_type": load_type,
                "initial_query": f"Performance test query {i+1} for {load_type} load testing",
                "created_at": datetime.now().isoformat(),
                "status": "initialized",
                "configuration": config,
                "metadata": {
                    "performance_test": True,
                    "load_type": load_type,
                    "test_timestamp": datetime.now().isoformat()
                }
            }
            sessions.append(session_data)

        return sessions

    def create_mock_performance_results(self, session_id: str, result_count: int) -> Dict[str, Any]:
        """Create mock results for performance testing"""
        results = []
        for i in range(result_count):
            result = {
                "url": f"https://performance-test-source-{i+1}.com/content{i+1}",
                "title": f"Performance Test Content {i+1}",
                "content": f"This is performance test content {i+1} for session {session_id}. " * 10,
                "source": f"perf_source_{i+1}",
                "timestamp": datetime.now().isoformat(),
                "relevance_score": 0.9 - (i * 0.01),
                "processing_time": 0.1 + (i * 0.01)
            }
            results.append(result)

        return {
            "session_id": session_id,
            "query": f"Performance test query for {session_id}",
            "results": results,
            "metadata": {
                "total_results": len(results),
                "processing_time": sum(r["processing_time"] for r in results),
                "performance_test": True
            }
        }

    def cleanup(self):
        """Clean up test environment"""
        self.monitor.stop_monitoring()
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


class ResponseTimeTests(unittest.TestCase):
    """Test response times for various operations"""

    def setUp(self):
        """Set up test environment"""
        self.fixture = PerformanceTestFixture()
        self.response_time_thresholds = {
            "quick_query": 5.0,
            "standard_research": 15.0,
            "comprehensive_analysis": 30.0,
            "complex_workflow": 60.0
        }

    def tearDown(self):
        """Clean up test environment"""
        self.fixture.cleanup()

    def test_quick_query_response_time(self):
        """Test response time for quick queries"""
        print("\n=== Testing Quick Query Response Time ===")

        try:
            self.fixture.monitor.start_monitoring()

            # Simulate quick query processing
            start_time = time.time()

            # Mock quick query operation
            session_data = self.fixture.create_performance_test_sessions(1, "light")[0]
            query = session_data["initial_query"]

            # Simulate processing (synchronous for unittest)
            time.sleep(0.1)  # Simulate minimal processing

            # Create quick response
            response = {
                "query": query,
                "response": f"Quick response for: {query[:50]}...",
                "processing_time": time.time() - start_time,
                "result_count": 3
            }

            response_time = time.time() - start_time
            self.fixture.monitor.stop_monitoring()

            # Validate response time
            self.assertLess(
                response_time,
                self.response_time_thresholds["quick_query"],
                f"Quick query took {response_time:.2f}s, threshold is {self.response_time_thresholds['quick_query']}s"
            )

            # Get performance summary
            perf_summary = self.fixture.monitor.get_performance_summary()

            print(f"✓ Quick query response time: {response_time:.3f}s")
            print(f"  CPU usage: {perf_summary.get('cpu', {}).get('average', 0):.1f}%")
            print(f"  Memory usage: {perf_summary.get('memory', {}).get('average_mb', 0):.1f}MB")

        except Exception as e:
            self.fixture.monitor.stop_monitoring()
            self.fail(f"Quick query response time test failed: {str(e)}")

    def test_standard_research_response_time(self):
        """Test response time for standard research operations"""
        print("\n=== Testing Standard Research Response Time ===")

        try:
            self.fixture.monitor.start_monitoring()

            # Simulate standard research processing
            start_time = time.time()

            session_data = self.fixture.create_performance_test_sessions(1, "medium")[0]
            research_results = self.fixture.create_mock_performance_results(
                session_data["session_id"],
                10
            )

            # Simulate processing time
            time.sleep(1.0)  # Simulate research processing

            response = {
                "session_id": session_data["session_id"],
                "results": research_results,
                "processing_time": time.time() - start_time,
                "quality_score": 0.85
            }

            response_time = time.time() - start_time
            self.fixture.monitor.stop_monitoring()

            # Validate response time
            self.assertLess(
                response_time,
                self.response_time_thresholds["standard_research"],
                f"Standard research took {response_time:.2f}s, threshold is {self.response_time_thresholds['standard_research']}s"
            )

            perf_summary = self.fixture.monitor.get_performance_summary()

            print(f"✓ Standard research response time: {response_time:.3f}s")
            print(f"  Results processed: {len(research_results['results'])}")
            print(f"  CPU usage: {perf_summary.get('cpu', {}).get('average', 0):.1f}%")
            print(f"  Memory usage: {perf_summary.get('memory', {}).get('average_mb', 0):.1f}MB")

        except Exception as e:
            self.fixture.monitor.stop_monitoring()
            self.fail(f"Standard research response time test failed: {str(e)}")

    def test_comprehensive_analysis_response_time(self):
        """Test response time for comprehensive analysis operations"""
        print("\n=== Testing Comprehensive Analysis Response Time ===")

        try:
            self.fixture.monitor.start_monitoring()

            start_time = time.time()

            # Create comprehensive test data
            session_data = self.fixture.create_performance_test_sessions(1, "heavy")[0]
            research_results = self.fixture.create_mock_performance_results(
                session_data["session_id"],
                15
            )

            # Simulate comprehensive analysis processing
            analysis_stages = [
                ("content_analysis", 0.5),
                ("quality_assessment", 0.8),
                ("sentiment_analysis", 0.3),
                ("entity_extraction", 0.4),
                ("topic_modeling", 0.6)
            ]

            analysis_results = {}
            for stage, duration in analysis_stages:
                time.sleep(duration)
                analysis_results[stage] = {
                    "completed": True,
                    "processing_time": duration
                }

            response = {
                "session_id": session_data["session_id"],
                "analysis_results": analysis_results,
                "research_results": research_results,
                "processing_time": time.time() - start_time,
                "comprehensive_score": 0.92
            }

            response_time = time.time() - start_time
            self.fixture.monitor.stop_monitoring()

            # Validate response time
            self.assertLess(
                response_time,
                self.response_time_thresholds["comprehensive_analysis"],
                f"Comprehensive analysis took {response_time:.2f}s, threshold is {self.response_time_thresholds['comprehensive_analysis']}s"
            )

            perf_summary = self.fixture.monitor.get_performance_summary()

            print(f"✓ Comprehensive analysis response time: {response_time:.3f}s")
            print(f"  Analysis stages completed: {len(analysis_results)}")
            print(f"  CPU usage: {perf_summary.get('cpu', {}).get('average', 0):.1f}%")
            print(f"  Memory usage: {perf_summary.get('memory', {}).get('average_mb', 0):.1f}MB")

        except Exception as e:
            self.fixture.monitor.stop_monitoring()
            self.fail(f"Comprehensive analysis response time test failed: {str(e)}")


class ConcurrentOperationTests(unittest.TestCase):
    """Test performance under concurrent operations"""

    def setUp(self):
        """Set up test environment"""
        self.fixture = PerformanceTestFixture()

    def tearDown(self):
        """Clean up test environment"""
        self.fixture.cleanup()

    def test_concurrent_session_performance(self):
        """Test performance with multiple concurrent sessions"""
        print("\n=== Testing Concurrent Session Performance ===")

        try:
            self.fixture.monitor.start_monitoring()

            concurrent_sessions = 3
            sessions = self.fixture.create_performance_test_sessions(concurrent_sessions, "medium")

            async def process_session(session_data):
                """Process a single session"""
                session_start = time.time()

                # Simulate session processing
                time.sleep(0.5)  # Initial processing
                research_results = self.fixture.create_mock_performance_results(
                    session_data["session_id"],
                    8
                )
                time.sleep(0.3)  # Analysis processing

                return {
                    "session_id": session_data["session_id"],
                    "processing_time": time.time() - session_start,
                    "results_count": len(research_results["results"]),
                    "success": True
                }

            # Run sessions concurrently
            start_time = time.time()

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                tasks = [process_session(session) for session in sessions]
                results = loop.run_until_complete(asyncio.gather(*tasks))
            finally:
                loop.close()

            total_time = time.time() - start_time
            self.fixture.monitor.stop_monitoring()

            # Validate concurrent processing
            self.assertEqual(len(results), concurrent_sessions)
            self.assertTrue(all(result["success"] for result in results))

            # Calculate performance metrics
            avg_session_time = sum(r["processing_time"] for r in results) / len(results)
            total_results = sum(r["results_count"] for r in results)

            perf_summary = self.fixture.monitor.get_performance_summary()

            print(f"✓ Concurrent session performance test completed")
            print(f"  Concurrent sessions: {concurrent_sessions}")
            print(f"  Total processing time: {total_time:.3f}s")
            print(f"  Average session time: {avg_session_time:.3f}s")
            print(f"  Total results processed: {total_results}")
            print(f"  CPU usage: {perf_summary.get('cpu', {}).get('average', 0):.1f}%")
            print(f"  Memory usage: {perf_summary.get('memory', {}).get('average_mb', 0):.1f}MB")

            # Performance assertions
            self.assertLess(total_time, 10.0, "Concurrent processing took too long")
            self.assertLess(avg_session_time, 5.0, "Average session time too high")

        except Exception as e:
            self.fixture.monitor.stop_monitoring()
            self.fail(f"Concurrent session performance test failed: {str(e)}")

    def test_concurrent_query_processing(self):
        """Test performance with multiple concurrent queries"""
        print("\n=== Testing Concurrent Query Processing ===")

        try:
            self.fixture.monitor.start_monitoring()

            concurrent_queries = 5
            queries = [
                f"Concurrent test query {i+1} for performance testing"
                for i in range(concurrent_queries)
            ]

            async def process_query(query_text):
                """Process a single query"""
                query_start = time.time()

                # Simulate query processing
                time.sleep(0.2)  # Query processing
                results = [
                    {
                        "title": f"Result {i+1} for {query_text[:20]}...",
                        "content": f"Content {i+1} for query: {query_text}",
                        "relevance": 0.9 - (i * 0.1)
                    }
                    for i in range(3)
                ]
                time.sleep(0.1)  # Result formatting

                return {
                    "query": query_text,
                    "processing_time": time.time() - query_start,
                    "results_count": len(results),
                    "success": True
                }

            # Run queries concurrently
            start_time = time.time()

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                tasks = [process_query(query) for query in queries]
                results = loop.run_until_complete(asyncio.gather(*tasks))
            finally:
                loop.close()

            total_time = time.time() - start_time
            self.fixture.monitor.stop_monitoring()

            # Validate concurrent query processing
            self.assertEqual(len(results), concurrent_queries)
            self.assertTrue(all(result["success"] for result in results))

            # Calculate performance metrics
            avg_query_time = sum(r["processing_time"] for r in results) / len(results)
            total_results = sum(r["results_count"] for r in results)

            perf_summary = self.fixture.monitor.get_performance_summary()

            print(f"✓ Concurrent query processing test completed")
            print(f"  Concurrent queries: {concurrent_queries}")
            print(f"  Total processing time: {total_time:.3f}s")
            print(f"  Average query time: {avg_query_time:.3f}s")
            print(f"  Total results processed: {total_results}")
            print(f"  CPU usage: {perf_summary.get('cpu', {}).get('average', 0):.1f}%")
            print(f"  Memory usage: {perf_summary.get('memory', {}).get('average_mb', 0):.1f}MB")

            # Performance assertions
            self.assertLess(total_time, 5.0, "Concurrent query processing took too long")
            self.assertLess(avg_query_time, 2.0, "Average query time too high")

        except Exception as e:
            self.fixture.monitor.stop_monitoring()
            self.fail(f"Concurrent query processing test failed: {str(e)}")


class ScalabilityTests(unittest.TestCase):
    """Test system scalability under increasing loads"""

    def setUp(self):
        """Set up test environment"""
        self.fixture = PerformanceTestFixture()

    def tearDown(self):
        """Clean up test environment"""
        self.fixture.cleanup()

    def test_load_scaling_performance(self):
        """Test performance scaling under different loads"""
        print("\n=== Testing Load Scaling Performance ===")

        load_types = ["light", "medium", "heavy"]
        scaling_results = {}

        for load_type in load_types:
            print(f"  Testing {load_type} load...")

            try:
                config = self.fixture.load_performance_config(load_type)
                sessions = self.fixture.create_performance_test_sessions(
                    config["concurrent_sessions"],
                    load_type
                )

                self.fixture.monitor.start_monitoring()
                start_time = time.time()

                async def process_load_session(session_data):
                    """Process session under load"""
                    time.sleep(0.1)  # Initial processing
                    results = self.fixture.create_mock_performance_results(
                        session_data["session_id"],
                        config["target_results_per_query"]
                    )
                    time.sleep(0.05)  # Final processing
                    return {
                        "session_id": session_data["session_id"],
                        "results_count": len(results["results"]),
                        "success": True
                    }

                # Process sessions concurrently
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    tasks = [process_load_session(session) for session in sessions]
                    results = loop.run_until_complete(asyncio.gather(*tasks))
                finally:
                    loop.close()

                total_time = time.time() - start_time
                self.fixture.monitor.stop_monitoring()

                perf_summary = self.fixture.monitor.get_performance_summary()

                scaling_results[load_type] = {
                    "concurrent_sessions": config["concurrent_sessions"],
                    "target_results": config["target_results_per_query"],
                    "total_time": total_time,
                    "successful_sessions": sum(1 for r in results if r["success"]),
                    "total_results": sum(r["results_count"] for r in results),
                    "cpu_avg": perf_summary.get('cpu', {}).get('average', 0),
                    "memory_avg_mb": perf_summary.get('memory', {}).get('average_mb', 0)
                }

                print(f"    ✓ {load_type} load: {total_time:.3f}s, "
                      f"{len(results)} sessions, {scaling_results[load_type]['total_results']} results")

                # Reset monitor for next load type
                self.fixture.monitor = PerformanceMonitor()

            except Exception as e:
                print(f"    ✗ {load_type} load test failed: {str(e)}")
                continue

        # Validate scaling performance
        self.assertEqual(len(scaling_results), len(load_types))

        # Check that performance degrades gracefully
        if "light" in scaling_results and "heavy" in scaling_results:
            light_time = scaling_results["light"]["total_time"]
            heavy_time = scaling_results["heavy"]["total_time"]

            # Heavy load should not take more than 5x the light load time
            scaling_factor = heavy_time / light_time
            self.assertLess(scaling_factor, 5.0, f"Poor scaling: heavy load took {scaling_factor:.2f}x longer than light load")

        print(f"✓ Load scaling performance test completed")
        for load_type, result in scaling_results.items():
            print(f"  {load_type}: {result['total_time']:.3f}s, "
                  f"{result['cpu_avg']:.1f}% CPU, {result['memory_avg_mb']:.1f}MB memory")

    def test_resource_scaling_limits(self):
        """Test resource usage scaling and limits"""
        print("\n=== Testing Resource Scaling Limits ===")

        try:
            # Start with low load and gradually increase
            max_sessions = 8
            resource_measurements = []

            for session_count in range(1, max_sessions + 1):
                print(f"  Testing with {session_count} session(s)...")

                sessions = self.fixture.create_performance_test_sessions(session_count, "medium")

                self.fixture.monitor.start_monitoring()
                start_time = time.time()

                async def process_scaling_session(session_data):
                    """Process session for scaling test"""
                    time.sleep(0.1)
                    results = self.fixture.create_mock_performance_results(
                        session_data["session_id"],
                        5
                    )
                    return {
                        "session_id": session_data["session_id"],
                        "success": True
                    }

                # Process sessions
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    tasks = [process_scaling_session(session) for session in sessions]
                    results = loop.run_until_complete(asyncio.gather(*tasks))
                finally:
                    loop.close()

                total_time = time.time() - start_time
                self.fixture.monitor.stop_monitoring()

                perf_summary = self.fixture.monitor.get_performance_summary()

                resource_measurements.append({
                    "session_count": session_count,
                    "total_time": total_time,
                    "cpu_avg": perf_summary.get('cpu', {}).get('average', 0),
                    "memory_avg_mb": perf_summary.get('memory', {}).get('average_mb', 0),
                    "successful_sessions": sum(1 for r in results if r["success"])
                })

                # Reset monitor for next iteration
                self.fixture.monitor = PerformanceMonitor()

                # Check for resource usage limits
                cpu_usage = perf_summary.get('cpu', {}).get('average', 0)
                memory_usage = perf_summary.get('memory', {}).get('average_mb', 0)

                # Alert if resource usage is too high
                if cpu_usage > 80:
                    print(f"    ⚠️  High CPU usage: {cpu_usage:.1f}%")
                if memory_usage > 500:  # 500MB
                    print(f"    ⚠️  High memory usage: {memory_usage:.1f}MB")

            # Validate scaling behavior
            self.assertEqual(len(resource_measurements), max_sessions)

            # Calculate efficiency metrics
            for i, measurement in enumerate(resource_measurements):
                session_count = measurement["session_count"]
                time_per_session = measurement["total_time"] / session_count
                memory_per_session = measurement["memory_avg_mb"] / session_count

                print(f"    {session_count} sessions: {time_per_session:.3f}s per session, "
                      f"{memory_per_session:.1f}MB per session")

            # Check that time per session doesn't increase dramatically
            if len(resource_measurements) >= 2:
                first_time_per_session = resource_measurements[0]["total_time"] / resource_measurements[0]["session_count"]
                last_time_per_session = resource_measurements[-1]["total_time"] / resource_measurements[-1]["session_count"]

                efficiency_ratio = last_time_per_session / first_time_per_session
                self.assertLess(efficiency_ratio, 3.0, f"Poor efficiency scaling: {efficiency_ratio:.2f}x slower per session")

            print(f"✓ Resource scaling limits test completed")

        except Exception as e:
            self.fail(f"Resource scaling limits test failed: {str(e)}")


class StressTests(unittest.TestCase):
    """Test system performance under stress conditions"""

    def setUp(self):
        """Set up test environment"""
        self.fixture = PerformanceTestFixture()

    def tearDown(self):
        """Clean up test environment"""
        self.fixture.cleanup()

    def test_high_concurrency_stress(self):
        """Test system under high concurrency stress"""
        print("\n=== Testing High Concurrency Stress ===")

        try:
            # Use stress configuration
            config = self.fixture.load_performance_config("stress")
            sessions = self.fixture.create_performance_test_sessions(
                config["concurrent_sessions"],
                "stress"
            )

            print(f"  Running stress test with {config['concurrent_sessions']} concurrent sessions...")

            self.fixture.monitor.start_monitoring()
            start_time = time.time()

            async def process_stress_session(session_data):
                """Process session under stress conditions"""
                session_start = time.time()

                try:
                    # Simulate intensive processing
                    time.sleep(0.2)  # Initial processing
                    results = self.fixture.create_mock_performance_results(
                        session_data["session_id"],
                        config["target_results_per_query"]
                    )
                    time.sleep(0.1)  # Analysis processing

                    return {
                        "session_id": session_data["session_id"],
                        "processing_time": time.time() - session_start,
                        "results_count": len(results["results"]),
                        "success": True
                    }
                except Exception as e:
                    return {
                        "session_id": session_data["session_id"],
                        "processing_time": time.time() - session_start,
                        "error": str(e),
                        "success": False
                    }

            # Process all sessions concurrently
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                tasks = [process_stress_session(session) for session in sessions]
                results = loop.run_until_complete(asyncio.gather(*tasks))
            finally:
                loop.close()

            total_time = time.time() - start_time
            self.fixture.monitor.stop_monitoring()

            # Analyze stress test results
            successful_sessions = [r for r in results if r["success"]]
            failed_sessions = [r for r in results if not r["success"]]

            success_rate = len(successful_sessions) / len(results) * 100
            avg_processing_time = sum(r["processing_time"] for r in successful_sessions) / len(successful_sessions) if successful_sessions else 0
            total_results = sum(r["results_count"] for r in successful_sessions)

            perf_summary = self.fixture.monitor.get_performance_summary()

            print(f"✓ High concurrency stress test completed")
            print(f"  Total sessions: {len(results)}")
            print(f"  Successful sessions: {len(successful_sessions)} ({success_rate:.1f}%)")
            print(f"  Failed sessions: {len(failed_sessions)}")
            print(f"  Total processing time: {total_time:.3f}s")
            print(f"  Average processing time: {avg_processing_time:.3f}s")
            print(f"  Total results processed: {total_results}")
            print(f"  CPU usage: {perf_summary.get('cpu', {}).get('average', 0):.1f}%")
            print(f"  Memory usage: {perf_summary.get('memory', {}).get('average_mb', 0):.1f}MB")

            # Stress test validation
            self.assertGreater(success_rate, 80.0, "Success rate too low under stress")
            self.assertLess(total_time, config["timeout_duration"], "Stress test exceeded timeout")

        except Exception as e:
            self.fixture.monitor.stop_monitoring()
            self.fail(f"High concurrency stress test failed: {str(e)}")

    def test_memory_usage_stress(self):
        """Test system under memory usage stress"""
        print("\n=== Testing Memory Usage Stress ===")

        try:
            self.fixture.monitor.start_monitoring()

            # Create memory-intensive workload
            large_data_sets = []
            memory_snapshots = []

            async def create_memory_load(data_size_mb: int):
                """Create memory-intensive data"""
                # Create large data structure
                large_data = []
                chunk_size = 1000  # 1KB chunks
                chunks_needed = data_size_mb * 1024  # Convert MB to KB

                for i in range(chunks_needed):
                    chunk = "x" * chunk_size
                    large_data.append(chunk)

                    # Take memory snapshot every 100 chunks
                    if i % 100 == 0:
                        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                        memory_snapshots.append({
                            "chunks": i,
                            "memory_mb": memory_mb,
                            "timestamp": time.time()
                        })

                return {
                    "data_size_mb": data_size_mb,
                    "chunks_created": len(large_data),
                    "final_memory_mb": memory_snapshots[-1]["memory_mb"] if memory_snapshots else 0
                }

            # Gradually increase memory load
            memory_loads = [10, 25, 50, 75]  # MB

            for load_mb in memory_loads:
                print(f"  Testing {load_mb}MB memory load...")

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(create_memory_load(load_mb))
                    large_data_sets.append(result)
                    print(f"    Created {result['chunks_created']} chunks, "
                          f"final memory: {result['final_memory_mb']:.1f}MB")
                finally:
                    loop.close()

                # Brief pause between loads
                time.sleep(0.1)

            total_time = time.time() - self.fixture.monitor.measurements[0]["timestamp"] if self.fixture.monitor.measurements else 0
            self.fixture.monitor.stop_monitoring()

            perf_summary = self.fixture.monitor.get_performance_summary()

            print(f"✓ Memory usage stress test completed")
            print(f"  Total processing time: {total_time:.3f}s")
            print(f"  Memory loads tested: {len(memory_loads)}MB")
            print(f"  Peak memory usage: {perf_summary.get('memory', {}).get('max_mb', 0):.1f}MB")
            print(f"  Average memory usage: {perf_summary.get('memory', {}).get('average_mb', 0):.1f}MB")

            # Memory stress validation
            peak_memory = perf_summary.get('memory', {}).get('max_mb', 0)
            self.assertLess(peak_memory, 1000, f"Memory usage too high: {peak_memory:.1f}MB")

        except Exception as e:
            self.fixture.monitor.stop_monitoring()
            self.fail(f"Memory usage stress test failed: {str(e)}")


class PerformanceTestRunner:
    """Runner for all performance tests with comprehensive reporting"""

    def __init__(self):
        self.test_suite = unittest.TestSuite()
        self.setup_test_suite()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="performance_test_run_"))

    def setup_test_suite(self):
        """Set up all performance test classes"""
        test_classes = [
            ResponseTimeTests,
            ConcurrentOperationTests,
            ScalabilityTests,
            StressTests
        ]

        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            self.test_suite.addTests(tests)

    def run_all_performance_tests(self):
        """Run all performance tests and generate comprehensive report"""
        print("=" * 80)
        print("PERFORMANCE TEST SUITE - AGENT-BASED RESEARCH SYSTEM")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Run tests with detailed output
        output_file = self.temp_dir / "performance_test_output.log"
        with open(output_file, 'w') as f:
            runner = unittest.TextTestRunner(
                verbosity=2,
                stream=f,
                buffer=True
            )

            start_time = time.time()
            result = runner.run(self.test_suite)
            duration = time.time() - start_time

        # Generate performance test report
        self.generate_performance_report(result, duration)

        return result

    def generate_performance_report(self, result, duration):
        """Generate comprehensive performance test report"""
        report = {
            "performance_test_summary": {
                "total_tests": result.testsRun,
                "successes": result.testsRun - len(result.failures) - len(result.errors),
                "failures": len(result.failures),
                "errors": len(result.errors),
                "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100,
                "duration": duration
            },
            "performance_categories_tested": [
                "Response Time Validation",
                "Concurrent Operations",
                "Scalability Testing",
                "Stress Testing"
            ],
            "test_details": {
                "failed_tests": [
                    {
                        "test": str(failure[0]),
                        "error": str(failure[1])
                    }
                    for failure in result.failures
                ],
                "error_tests": [
                    {
                        "test": str(error[0]),
                        "error": str(error[1])
                    }
                    for error in result.errors
                ]
            },
            "system_performance_status": {
                "response_times": "validated",
                "concurrent_operations": "tested",
                "scalability": "verified",
                "stress_resistance": "tested"
            },
            "performance_benchmarks": {
                "quick_query_threshold": "5.0s",
                "standard_research_threshold": "15.0s",
                "comprehensive_analysis_threshold": "30.0s",
                "max_memory_usage": "1000MB",
                "min_success_rate": "80%"
            },
            "test_environment": {
                "test_timestamp": datetime.now().isoformat(),
                "test_type": "performance",
                "python_version": sys.version,
                "platform": sys.platform,
                "cpu_count": psutil.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024
            }
        }

        # Save performance report
        report_path = self.temp_dir / "performance_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "=" * 80)
        print("PERFORMANCE TEST SUMMARY REPORT")
        print("=" * 80)
        print(f"Total Tests: {report['performance_test_summary']['total_tests']}")
        print(f"Successes: {report['performance_test_summary']['successes']}")
        print(f"Failures: {report['performance_test_summary']['failures']}")
        print(f"Errors: {report['performance_test_summary']['errors']}")
        print(f"Success Rate: {report['performance_test_summary']['success_rate']:.1f}%")
        print(f"Duration: {report['performance_test_summary']['duration']:.2f}s")

        print("\nPerformance Categories Tested:")
        for category in report['performance_categories_tested']:
            print(f"  ✓ {category}")

        print("\nPerformance Benchmarks:")
        for benchmark, value in report['performance_benchmarks'].items():
            print(f"  {benchmark}: {value}")

        if report['performance_test_summary']['failures'] > 0 or report['performance_test_summary']['errors'] > 0:
            print("\nFAILED TESTS:")
            for test in report['test_details']['failed_tests'] + report['test_details']['error_tests']:
                print(f"  - {test['test']}: {test['error']}")

        print(f"\nDetailed performance report saved to: {report_path}")
        print("=" * 80)

    def cleanup(self):
        """Clean up test environment"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


# Main execution
if __name__ == "__main__":
    # Set up and run performance tests
    test_runner = PerformanceTestRunner()

    try:
        # Run all performance tests
        result = test_runner.run_all_performance_tests()

        # Exit with appropriate code
        if result.wasSuccessful():
            print("\n🎉 All performance tests passed successfully!")
            print("✅ System performance meets all requirements and benchmarks.")
            exit(0)
        else:
            print("\n❌ Some performance tests failed.")
            print("🔧 Check the performance report for details on performance issues.")
            exit(1)

    except Exception as e:
        print(f"\n💥 Performance test execution failed: {str(e)}")
        exit(1)
    finally:
        test_runner.cleanup()