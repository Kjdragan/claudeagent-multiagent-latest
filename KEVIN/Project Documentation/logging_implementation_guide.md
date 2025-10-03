# Logging System Implementation Guide

## Step-by-Step Implementation

This guide provides step-by-step instructions for implementing the comprehensive logging system in your multi-agent applications.

## Prerequisites

### Required Dependencies
```bash
# Core dependencies (always required)
pip install asyncio pathlib dataclasses datetime typing json

# Optional dependencies for advanced monitoring
pip install streamlit pandas numpy plotly
```

### Project Structure
Ensure your project has the following structure:
```
your_project/
├── multi_agent_research_system/     # Logging system files
├── your_agent_code/
└── logs/                           # Will be created automatically
```

## Step 1: Basic Setup

### 1.1 Initialize Logging System
```python
# main.py or app.py
from multi_agent_research_system.monitoring_integration import MonitoringIntegration

async def main():
    # Initialize monitoring with session ID
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    monitoring = MonitoringIntegration(
        session_id=session_id,
        monitoring_dir="monitoring",
        enable_advanced_monitoring=True  # Set to False for basic logging only
    )

    print(f"✅ Logging system initialized with session: {session_id}")
    return monitoring
```

### 1.2 Configure Environment (Optional)
```python
import os

# Set environment variables (optional, can also be set in .env file)
os.environ['LOG_LEVEL'] = 'INFO'
os.environ['LOG_DIR'] = 'logs'
os.environ['MONITORING_DIR'] = 'monitoring'
os.environ['ENABLE_ADVANCED_MONITORING'] = 'true'
os.environ['LOG_RETENTION_DAYS'] = '30'
```

## Step 2: SDK Hook Integration

### 2.1 Register Hooks with ClaudeSDKClient
```python
from multi_agent_research_system.agent_logging import HookIntegration
from claude_agent_sdk import ClaudeSDKClient

async def setup_client_with_hooks():
    # Create ClaudeSDKClient
    client = ClaudeSDKClient()

    # Create hook integration
    hooks = HookIntegration(session_id="your_session_id")

    # Register hooks
    await client.register_hooks([hooks])

    print("✅ SDK hooks registered - all agent activities will be logged")
    return client
```

### 2.2 Verify Hook Registration
```python
# Test that hooks are working
async def test_hooks(client):
    # This should automatically trigger hook logging
    response = await client.send_message("Hello, test message")

    # Check if logs were created
    import os
    if os.path.exists("logs"):
        log_files = os.listdir("logs")
        print(f"✅ Found {len(log_files)} log files")
    else:
        print("⚠️  No log directory found")
```

## Step 3: Agent-Specific Logging

### 3.1 Create Agent Loggers
```python
from multi_agent_research_system.agent_logging import AgentLoggerFactory

def create_agent_loggers(session_id):
    """Create specialized loggers for each agent type"""

    loggers = {}

    # Research Agent Logger
    loggers['research'] = AgentLoggerFactory.create_logger(
        agent_type="research_agent",
        session_id=session_id
    )

    # Report Agent Logger
    loggers['report'] = AgentLoggerFactory.create_logger(
        agent_type="report_agent",
        session_id=session_id
    )

    # Editor Agent Logger
    loggers['editor'] = AgentLoggerFactory.create_logger(
        agent_type="editor_agent",
        session_id=session_id
    )

    # UI Coordinator Logger
    loggers['ui'] = AgentLoggerFactory.create_logger(
        agent_type="ui_coordinator",
        session_id=session_id
    )

    print("✅ Agent loggers created")
    return loggers
```

### 3.2 Use Agent Loggers in Your Code
```python
# Example: Research Agent Implementation
class ResearchAgent:
    def __init__(self, session_id, logger):
        self.session_id = session_id
        self.logger = logger

    async def perform_research(self, query):
        # Log research start
        self.logger.info(
            message="Starting research",
            metadata={
                "query": query,
                "agent": "research_agent",
                "action": "research_start"
            }
        )

        try:
            # Perform research logic here
            results = await self._search_sources(query)

            # Log successful research
            self.logger.info(
                message="Research completed",
                metadata={
                    "query": query,
                    "sources_found": len(results),
                    "agent": "research_agent",
                    "action": "research_complete"
                }
            )

            return results

        except Exception as e:
            # Log error
            self.logger.error(
                message="Research failed",
                metadata={
                    "query": query,
                    "error": str(e),
                    "agent": "research_agent",
                    "action": "research_error"
                }
            )
            raise
```

## Step 4: Performance Monitoring

### 4.1 Monitor Tool Execution
```python
from multi_agent_research_system.monitoring_integration import MonitoringIntegration

class ToolManager:
    def __init__(self, monitoring):
        self.monitoring = monitoring

    async def execute_tool(self, tool_name, agent_name, tool_function, *args, **kwargs):
        """Execute a tool with automatic performance monitoring"""

        async with self.monitoring.monitor_tool_execution(tool_name, agent_name):
            # Execute the actual tool
            result = await tool_function(*args, **kwargs)

            # Log additional metrics
            self.monitoring.log_agent_activity(
                agent_name=agent_name,
                activity=f"tool_{tool_name}_completed",
                metadata={
                    "tool_name": tool_name,
                    "input_size": len(str(args)),
                    "success": True
                }
            )

            return result
```

### 4.2 Monitor Custom Workflows
```python
async def run_research_workflow(monitoring, query):
    """Example of monitoring a complete workflow"""

    with monitoring.monitor_workflow("research_pipeline"):
        # Step 1: Research
        research_logger = monitoring.get_agent_logger("research_agent")
        research_logger.info(f"Starting research for: {query}")

        # Step 2: Analysis
        with monitoring.monitor_workflow("analysis_phase"):
            analysis_results = await analyze_data(query)

        # Step 3: Report Generation
        report_logger = monitoring.get_agent_logger("report_agent")
        report_logger.info("Generating report")

        final_report = await generate_report(analysis_results)

        return final_report
```

## Step 5: Advanced Features Setup

### 5.1 Enable Real-time Dashboard (Optional)
```python
# dashboard.py - Run separately or integrate into your app
import streamlit as st
from multi_agent_research_system.monitoring.real_time_dashboard import RealTimeDashboard

def run_dashboard():
    dashboard = RealTimeDashboard(monitoring_dir="monitoring")
    dashboard.run()

# To run: streamlit run dashboard.py
```

### 5.2 Set Up Log Analysis
```python
from multi_agent_research_system.log_analysis import (
    LogAggregator,
    LogSearchEngine,
    AnalyticsEngine,
    ReportGenerator
)

async def setup_log_analysis(session_id):
    """Set up log analysis and reporting"""

    # Initialize components
    aggregator = LogAggregator(session_id=session_id)
    search_engine = LogSearchEngine()
    analytics = AnalyticsEngine(session_id=session_id)
    report_gen = ReportGenerator(session_id=session_id)

    # Aggregate existing logs
    await aggregator.aggregate_logs()

    # Build search index
    entries = aggregator.get_entries()
    search_engine.build_index(entries)

    print("✅ Log analysis system ready")
    return aggregator, search_engine, analytics, report_gen
```

### 5.3 Generate Reports
```python
from datetime import datetime, timedelta
from multi_agent_research_system.log_analysis.report_generator import ReportType, ReportFormat

async def generate_daily_report(session_id):
    """Generate daily performance report"""

    report_gen = ReportGenerator(session_id=session_id)

    # Generate daily summary
    report = await report_gen.generate_report(
        report_type=ReportType.DAILY_SUMMARY,
        period_start=datetime.now() - timedelta(days=1),
        period_end=datetime.now(),
        formats=[ReportFormat.HTML, ReportFormat.JSON]
    )

    print(f"✅ Daily report generated: {report.report_id}")
    print(f"📄 Report files: {list(report.file_paths.values())}")

    return report
```

## Step 6: Error Handling and Troubleshooting

### 6.1 Graceful Error Handling
```python
async def safe_monitoring_setup(session_id):
    """Set up monitoring with error handling"""

    try:
        # Try to initialize with advanced monitoring
        monitoring = MonitoringIntegration(
            session_id=session_id,
            enable_advanced_monitoring=True
        )

        if monitoring.advanced_monitoring_enabled:
            print("✅ Advanced monitoring enabled")
        else:
            print("⚠️  Advanced monitoring not available, using basic logging")

        return monitoring

    except Exception as e:
        print(f"❌ Monitoring setup failed: {e}")

        # Fallback to basic logging
        try:
            monitoring = MonitoringIntegration(
                session_id=session_id,
                enable_advanced_monitoring=False
            )
            print("✅ Basic logging enabled as fallback")
            return monitoring
        except Exception as fallback_error:
            print(f"❌ Even basic logging failed: {fallback_error}")
            raise
```

### 6.2 Health Check Function
```python
def check_system_health(monitoring):
    """Check the health of the logging system"""

    health_status = {
        "basic_logging": monitoring.basic_logging_enabled,
        "advanced_monitoring": monitoring.advanced_monitoring_enabled,
        "log_directory_exists": os.path.exists("logs"),
        "monitoring_directory_exists": os.path.exists("monitoring"),
    }

    if monitoring.advanced_monitoring_enabled:
        health_status.update({
            "system_health": monitoring.get_system_health(),
            "metrics_available": monitoring.metrics_collector is not None,
            "dashboard_available": monitoring.real_time_dashboard is not None
        })

    return health_status
```

## Step 7: Production Deployment

### 7.1 Production Configuration
```python
# production_config.py
import os

def configure_for_production():
    """Configure logging system for production environment"""

    # Production settings
    os.environ['LOG_LEVEL'] = 'INFO'  # No DEBUG in production
    os.environ['LOG_RETENTION_DAYS'] = '30'  # Keep logs for 30 days
    os.environ['ENABLE_ADVANCED_MONITORING'] = 'true'

    # Performance settings
    os.environ['METRICS_COLLECTION_INTERVAL'] = '60'  # Collect every minute
    os.environ['MAX_LOG_FILE_SIZE'] = '50MB'  # Larger files in production
    os.environ['BACKUP_COUNT'] = '10'  # Keep more backups

    print("✅ Production configuration applied")
```

### 7.2 Docker Integration
```dockerfile
# Dockerfile
FROM python:3.9

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your code
COPY . .

# Create log directories
RUN mkdir -p logs monitoring

# Set environment variables
ENV LOG_LEVEL=INFO
ENV LOG_RETENTION_DAYS=30
ENV ENABLE_ADVANCED_MONITORING=true

# Expose dashboard port
EXPOSE 8501

# Run your application
CMD ["python", "main.py"]
```

### 7.3 Monitoring Script
```python
# monitor.py - Run as separate process for monitoring
import asyncio
import time
from multi_agent_research_system.monitoring_integration import MonitoringIntegration

async def monitoring_daemon():
    """Background monitoring daemon"""

    monitoring = MonitoringIntegration(
        session_id="monitoring_daemon",
        enable_advanced_monitoring=True
    )

    while True:
        # Check system health
        health = monitoring.get_system_health()

        if health['status'] == 'critical':
            print(f"🚨 CRITICAL: System health is critical (score: {health['overall_score']})")
            # Send alert, notification, etc.

        # Log periodic status
        monitoring.log_agent_activity(
            agent_name="monitoring_daemon",
            activity="health_check",
            metadata=health
        )

        # Sleep for 5 minutes
        await asyncio.sleep(300)

if __name__ == "__main__":
    asyncio.run(monitoring_daemon())
```

## Step 8: Testing and Validation

### 8.1 Test Script
```python
# test_logging.py
import asyncio
from multi_agent_research_system.monitoring_integration import MonitoringIntegration

async def test_logging_system():
    """Test all logging system features"""

    session_id = "test_session"
    monitoring = MonitoringIntegration(session_id=session_id)

    print("🧪 Testing logging system...")

    # Test 1: Basic logging
    monitoring.log_agent_activity("test_agent", "test_activity")
    print("✅ Basic logging test passed")

    # Test 2: Agent-specific logging
    from multi_agent_research_system.agent_logging import AgentLoggerFactory
    logger = AgentLoggerFactory.create_logger("research_agent", session_id)
    logger.info("Test message from research agent")
    print("✅ Agent-specific logging test passed")

    # Test 3: Performance monitoring
    with monitoring.monitor_tool_execution("test_tool", "test_agent"):
        await asyncio.sleep(0.1)  # Simulate work
    print("✅ Performance monitoring test passed")

    # Test 4: System health
    health = monitoring.get_system_health()
    print(f"✅ System health: {health['status']} (score: {health['overall_score']})")

    # Test 5: Advanced features (if available)
    if monitoring.advanced_monitoring_enabled:
        metrics = monitoring.metrics_collector.get_current_metrics()
        print(f"✅ Advanced monitoring active - {len(metrics)} metrics available")
    else:
        print("⚠️  Advanced monitoring not available")

    print("🎉 All tests completed!")

if __name__ == "__main__":
    asyncio.run(test_logging_system())
```

### 8.2 Integration Test
```python
# integration_test.py
async def test_complete_workflow():
    """Test complete workflow with logging"""

    # Initialize
    session_id = f"integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    monitoring = MonitoringIntegration(session_id=session_id)

    # Simulate agent workflow
    print("🔄 Simulating agent workflow...")

    # Research phase
    with monitoring.monitor_workflow("research_phase"):
        await asyncio.sleep(0.5)
        monitoring.log_agent_activity("research_agent", "search_completed",
                                    metadata={"sources": 10})

    # Analysis phase
    with monitoring.monitor_workflow("analysis_phase"):
        await asyncio.sleep(0.3)
        monitoring.log_agent_activity("analysis_agent", "analysis_completed")

    # Report phase
    with monitoring.monitor_workflow("report_phase"):
        await asyncio.sleep(0.2)
        monitoring.log_agent_activity("report_agent", "report_generated",
                                    metadata={"pages": 5})

    # Check results
    health = monitoring.get_system_health()
    print(f"✅ Workflow completed - System health: {health['status']}")

    # Generate summary
    metrics = monitoring.metrics_collector.get_current_metrics()
    print(f"📊 Total operations: {metrics.get('total_operations', 0)}")
    print(f"⏱️  Average response time: {metrics.get('avg_response_time', 0):.2f}s")
```

## Complete Example

### Full Application Example
```python
# app.py - Complete example with all features
import asyncio
import os
from datetime import datetime
from multi_agent_research_system.monitoring_integration import MonitoringIntegration
from multi_agent_research_system.agent_logging import AgentLoggerFactory

class MultiAgentApp:
    def __init__(self):
        self.session_id = f"app_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.monitoring = None
        self.agent_loggers = {}

    async def initialize(self):
        """Initialize the complete logging system"""

        print(f"🚀 Initializing app with session: {self.session_id}")

        # 1. Setup monitoring
        self.monitoring = MonitoringIntegration(
            session_id=self.session_id,
            enable_advanced_monitoring=True
        )

        # 2. Create agent loggers
        self.agent_loggers = {
            'research': AgentLoggerFactory.create_logger("research_agent", self.session_id),
            'report': AgentLoggerFactory.create_logger("report_agent", self.session_id),
            'editor': AgentLoggerFactory.create_logger("editor_agent", self.session_id),
            'ui': AgentLoggerFactory.create_logger("ui_coordinator", self.session_id)
        }

        # 3. Setup log analysis
        from multi_agent_research_system.log_analysis import (
            LogAggregator, AnalyticsEngine, ReportGenerator
        )

        self.log_aggregator = LogAggregator(session_id=self.session_id)
        self.analytics = AnalyticsEngine(session_id=self.session_id)
        self.report_generator = ReportGenerator(session_id=self.session_id)

        print("✅ App initialization complete")

    async def run_demo_workflow(self):
        """Run a demo workflow to showcase logging"""

        print("🔄 Running demo workflow...")

        # Research Phase
        self.agent_loggers['research'].info("Starting market research")
        with self.monitoring.monitor_workflow("market_research"):
            await asyncio.sleep(1.0)
            self.agent_loggers['research'].info(
                "Research completed",
                metadata={"sources_analyzed": 25, "insights_found": 8}
            )

        # Analysis Phase
        self.agent_loggers['report'].info("Starting analysis")
        with self.monitoring.monitor_workflow("data_analysis"):
            await asyncio.sleep(0.5)
            self.agent_loggers['report'].info(
                "Analysis completed",
                metadata={"data_points": 150, "trends_identified": 5}
            )

        # Report Generation
        self.agent_loggers['editor'].info("Creating report")
        with self.monitoring.monitor_workflow("report_creation"):
            await asyncio.sleep(0.3)
            self.agent_loggers['editor'].info(
                "Report created",
                metadata={"sections": 6, "word_count": 2500}
            )

        print("✅ Demo workflow completed")

    async def generate_summary_report(self):
        """Generate a summary report of the session"""

        print("📊 Generating summary report...")

        # Aggregate logs
        await self.log_aggregator.aggregate_logs()

        # Generate analytics
        entries = self.log_aggregator.get_entries()
        analysis = await self.analytics.analyze_logs(entries, ['performance', 'usage'])

        # Generate report
        from multi_agent_research_system.log_analysis.report_generator import ReportType, ReportFormat
        from datetime import timedelta

        report = await self.report_generator.generate_report(
            report_type=ReportType.DAILY_SUMMARY,
            period_start=datetime.now() - timedelta(hours=1),
            period_end=datetime.now(),
            formats=[ReportFormat.HTML]
        )

        print(f"✅ Summary report generated: {report.report_id}")
        return report

    def print_session_summary(self):
        """Print a summary of the session"""

        print("\n" + "="*50)
        print("📈 SESSION SUMMARY")
        print("="*50)

        # System health
        health = self.monitoring.get_system_health()
        print(f"System Health: {health['status']} (Score: {health['overall_score']}/100)")

        # Metrics
        if self.monitoring.advanced_monitoring_enabled:
            metrics = self.monitoring.metrics_collector.get_current_metrics()
            print(f"Total Operations: {metrics.get('total_operations', 0)}")
            print(f"Active Agents: {metrics.get('active_agents', 0)}")
            print(f"Success Rate: {metrics.get('success_rate', 0):.1f}%")

        # Log files
        if os.path.exists("logs"):
            log_files = len([f for f in os.listdir("logs") if f.endswith('.log')])
            print(f"Log Files Created: {log_files}")

        print("="*50)

async def main():
    """Main application entry point"""

    app = MultiAgentApp()

    try:
        # Initialize
        await app.initialize()

        # Run demo workflow
        await app.run_demo_workflow()

        # Generate summary report
        await app.generate_summary_report()

        # Print session summary
        app.print_session_summary()

    except Exception as e:
        print(f"❌ Application error: {e}")
        import traceback
        traceback.print_exc()

    print("🎉 Application completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
```

## Next Steps

1. **Run the test suite** to verify everything works:
   ```bash
   cd multi_agent_research_system
   python test_log_analysis.py
   ```

2. **Start with basic logging** and gradually enable advanced features

3. **Monitor system performance** and adjust configuration as needed

4. **Set up regular reporting** for ongoing insights

5. **Configure alerts** for critical system events

6. **Document your specific configuration** for your team

---

This implementation guide provides everything you need to successfully integrate the comprehensive logging system into your multi-agent applications. Start with the basic setup and gradually enable advanced features as needed.