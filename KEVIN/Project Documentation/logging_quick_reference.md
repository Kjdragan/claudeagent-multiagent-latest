# Logging System Quick Reference Guide

## Quick Start

### 1. Basic Setup
```python
from monitoring_integration import MonitoringIntegration

# Initialize logging and monitoring
monitoring = MonitoringIntegration(
    session_id="my_session_123",
    enable_advanced_monitoring=True  # Set to False for basic logging only
)
```

### 2. Agent-Specific Logging
```python
from agent_logging import AgentLoggerFactory

# Create logger for your agent type
logger = AgentLoggerFactory.create_logger(
    agent_type="research_agent",  # or "report_agent", "editor_agent", "ui_coordinator"
    session_id="my_session_123"
)

# Log activities
logger.info("Starting research", metadata={"query": "AI trends"})
logger.error("Search failed", metadata={"error": "timeout"})
```

### 3. SDK Hook Integration
```python
from agent_logging import HookIntegration

# Create hook integration
hooks = HookIntegration(session_id="my_session_123")

# Register with ClaudeSDKClient
await client.register_hooks([hooks])
# All agent activities are now automatically logged!
```

### 4. Performance Monitoring
```python
# Monitor tool execution
async with monitoring.monitor_tool_execution("web_search", "research_agent"):
    results = await perform_search(query)

# Monitor custom activities
with monitoring.monitor_workflow("research_pipeline"):
    await run_research_pipeline()
```

## Log Levels
- **DEBUG**: Detailed debugging information
- **INFO**: General information about agent activities
- **WARNING**: Something unexpected happened, but system is working
- **ERROR**: Error occurred, but system can continue
- **CRITICAL**: Serious error, system may not be able to continue

## Agent Types and Loggers

### Research Agent
```python
logger = AgentLoggerFactory.create_logger("research_agent", session_id)
logger.log_research_activity(query="...", sources_found=10, execution_time=2.5)
```

### Report Agent
```python
logger = AgentLoggerFactory.create_logger("report_agent", session_id)
logger.log_report_generation(report_type="summary", pages=5, processing_time=1.2)
```

### Editor Agent
```python
logger = AgentLoggerFactory.create_logger("editor_agent", session_id)
logger.log_edit_operation(operation="insert", section="introduction", changes=3)
```

### UI Coordinator
```python
logger = AgentLoggerFactory.create_logger("ui_coordinator", session_id)
logger.log_ui_interaction(component="dashboard", action="click", user_id="user123")
```

## Monitoring Commands

### System Health
```python
health = monitoring.get_system_health()
print(f"System status: {health['status']}")  # healthy, warning, critical
print(f"Overall score: {health['overall_score']}/100")
```

### Metrics
```python
# Current metrics
metrics = monitoring.metrics_collector.get_current_metrics()
print(f"Active agents: {metrics['active_agents']}")
print(f"Tool executions: {metrics['total_tool_executions']}")

# Agent-specific metrics
agent_metrics = monitoring.metrics_collector.get_agent_metrics("research_agent")
print(f"Agent success rate: {agent_metrics['success_rate']}%")
```

### Performance
```python
# Get performance summary
perf_summary = monitoring.performance_monitor.get_performance_summary()
print(f"Avg response time: {perf_summary['avg_response_time']}s")
print(f"Total operations: {perf_summary['total_operations']}")
```

## Log Analysis

### Search Logs
```python
from log_analysis import LogAggregator, LogSearchEngine, SearchQuery

# Aggregate logs from all sources
aggregator = LogAggregator(session_id="my_session_123")
await aggregator.aggregate_logs()

# Build search index
search_engine = LogSearchEngine()
entries = aggregator.get_entries()
search_engine.build_index(entries)

# Search for errors
error_query = SearchQuery(
    field="level",
    operator=SearchOperator.EQUALS,
    value="ERROR"
)
results, stats = search_engine.search(entries, error_query)
print(f"Found {len(results)} errors in {stats.execution_time_ms}ms")
```

### Analytics
```python
from log_analysis import AnalyticsEngine

# Analyze logs
analytics = AnalyticsEngine(session_id="my_session_123")
analysis = await analytics.analyze_logs(entries, ['performance', 'errors', 'trends'])

# Get insights
insights = analysis['insights']
for insight in insights:
    print(f"- {insight['type']}: {insight['description']}")
```

### Compliance Reports
```python
from log_analysis import AuditTrailManager, ComplianceStandard

# Create audit trail
audit = AuditTrailManager(session_id="my_session_123")

# Generate compliance report
report = await audit.generate_compliance_report(
    standard=ComplianceStandard.GDPR,
    period_start=datetime.now() - timedelta(days=30),
    period_end=datetime.now()
)
print(f"Compliance score: {report.compliance_score}%")
```

## Report Generation

### Quick Reports
```python
from log_analysis import ReportGenerator, ReportType, ReportFormat

generator = ReportGenerator(session_id="my_session_123")

# Daily summary
daily_report = await generator.generate_report(
    report_type=ReportType.DAILY_SUMMARY,
    period_start=datetime.now() - timedelta(days=1),
    period_end=datetime.now(),
    formats=[ReportFormat.HTML]
)

# Performance analysis
perf_report = await generator.generate_report(
    report_type=ReportType.PERFORMANCE_ANALYSIS,
    period_start=datetime.now() - timedelta(hours=24),
    period_end=datetime.now(),
    formats=[ReportFormat.JSON, ReportFormat.HTML]
)
```

## Configuration

### Environment Variables
```bash
# Basic settings
LOG_LEVEL=INFO
LOG_DIR=logs
MONITORING_DIR=monitoring

# Advanced features
ENABLE_ADVANCED_MONITORING=true
LOG_RETENTION_DAYS=30

# Performance settings
METRICS_COLLECTION_INTERVAL=30  # seconds
MAX_LOG_FILE_SIZE=10MB
BACKUP_COUNT=5
```

### Code Configuration
```python
# Minimal setup (basic logging only)
monitoring = MonitoringIntegration(
    session_id="session_123",
    enable_advanced_monitoring=False
)

# Full setup (all features)
monitoring = MonitoringIntegration(
    session_id="session_123",
    monitoring_dir="custom_monitoring",
    enable_advanced_monitoring=True
)
```

## Troubleshooting

### Check if Logging is Working
```python
# Test basic logging
monitoring.log_agent_activity("test_agent", "test_activity")
print("✅ Basic logging works" if monitoring.basic_logging_enabled else "❌ Basic logging failed")

# Check advanced monitoring
if monitoring.advanced_monitoring_enabled:
    print("✅ Advanced monitoring enabled")
else:
    print("⚠️  Advanced monitoring disabled (optional dependencies missing)")
```

### Common Issues

#### No logs appearing
```python
# Check log directory
import os
print(f"Log directory exists: {os.path.exists('logs')}")

# Check session ID
print(f"Session ID: {monitoring.session_id}")

# Test log write
monitoring.log_agent_activity("test", "test message")
```

#### Advanced monitoring not working
```python
# Check dependencies
try:
    import streamlit
    print("✅ Streamlit available")
except ImportError:
    print("❌ Streamlit not installed - advanced monitoring disabled")

# Check monitoring flag
print(f"Advanced monitoring: {monitoring.enable_advanced_monitoring}")
print(f"Monitoring available: {monitoring.MONITORING_AVAILABLE}")
```

### Get System Status
```python
# Complete system health check
status = {
    "basic_logging": monitoring.basic_logging_enabled,
    "advanced_monitoring": monitoring.advanced_monitoring_enabled,
    "system_health": monitoring.get_system_health(),
    "active_sessions": len(monitoring.agent_loggers),
    "metrics_available": monitoring.metrics_collector is not None
}

print("System Status:")
for key, value in status.items():
    print(f"  {key}: {value}")
```

## Performance Tips

### For Production
1. **Use appropriate log levels** - Avoid DEBUG in production
2. **Set reasonable retention** - Don't keep logs forever
3. **Monitor disk space** - Logs can grow quickly
4. **Use structured fields** - Consistent field names improve search performance

### For Development
1. **Enable DEBUG logging** - Get detailed information
2. **Use shorter retention** - Save disk space during development
3. **Monitor performance** - Check overhead of logging
4. **Test all features** - Verify monitoring works as expected

### Memory Management
```python
# Configure retention (cleanup old logs)
monitoring.configure_retention(
    log_retention_days=7,      # Keep logs for 7 days
    metrics_retention_hours=24, # Keep metrics for 24 hours
    cleanup_interval_hours=6   # Run cleanup every 6 hours
)
```

## File Structure

```
multi_agent_research_system/
├── agent_logging/
│   ├── __init__.py
│   ├── structured_logger.py      # Core logging functionality
│   ├── hook_integration.py       # SDK hook integration
│   ├── agent_logger.py          # Agent-specific logging
│   └── agent_loggers/           # Specialized loggers
├── monitoring/
│   ├── __init__.py
│   ├── metrics_collector.py     # Real-time metrics
│   ├── performance_monitor.py   # Performance tracking
│   ├── system_health.py         # Health monitoring
│   ├── real_time_dashboard.py   # Streamlit dashboard
│   └── diagnostics.py           # Diagnostic tools
├── log_analysis/
│   ├── __init__.py
│   ├── log_aggregator.py        # Log collection
│   ├── log_search.py           # Search functionality
│   ├── analytics_engine.py     # Analytics and insights
│   ├── audit_trail.py          # Compliance tracking
│   └── report_generator.py     # Report generation
├── monitoring_integration.py    # Unified interface
└── test_log_analysis.py        # Test suite
```

## Next Steps

1. **Initialize monitoring** in your application startup
2. **Register hooks** with your ClaudeSDKClient instance
3. **Create agent-specific loggers** for each agent type
4. **Configure monitoring** based on your environment needs
5. **Set up log analysis** for compliance and insights
6. **Monitor system health** regularly

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the comprehensive documentation
3. Run the test suite to verify functionality
4. Check system health and configuration

---

**Remember**: The logging system is designed with graceful degradation. If advanced features aren't available, it will automatically fall back to basic logging functionality.