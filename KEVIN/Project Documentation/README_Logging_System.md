# Comprehensive Logging and Monitoring System

A production-ready 5-phase logging and monitoring system for multi-agent applications built on the Claude Agent SDK.

## 🚀 Quick Start

```python
from multi_agent_research_system.monitoring_integration import MonitoringIntegration

# Initialize in 1 line
monitoring = MonitoringIntegration(session_id="my_session", enable_advanced_monitoring=True)

# Your agents are now being logged automatically!
```

## 📚 Documentation

### 📖 [Complete System Documentation](comprehensive_logging_system.md)
- Full architecture overview
- Detailed component descriptions
- API reference
- Security and compliance features

### ⚡ [Quick Reference Guide](logging_quick_reference.md)
- Fast setup instructions
- Common usage patterns
- Troubleshooting tips
- Performance considerations

### 🔧 [Implementation Guide](logging_implementation_guide.md)
- Step-by-step setup
- Code examples
- Production deployment
- Testing and validation

## 🏗️ System Architecture

```
Phase 1: Structured Logging Framework     → Foundation
Phase 2: Comprehensive Hook System       → SDK Integration
Phase 3: Agent-Specific Logging          → Specialized Tracking
Phase 4: Advanced Monitoring             → Real-time Insights
Phase 5: Log Analysis & Reporting        → Analytics & Compliance
```

## ✨ Key Features

### 🔍 **Complete Visibility**
- Track all agent activities with structured JSON logging
- Automatic SDK hook integration
- Correlation IDs for request tracing
- Multi-source log aggregation

### 📊 **Real-time Monitoring**
- Live performance metrics
- System health monitoring
- Interactive Streamlit dashboard
- Alerting and anomaly detection

### 🔬 **Advanced Analytics**
- Automated log analysis
- Trend detection and insights
- Performance bottleneck identification
- Usage pattern analysis

### 🛡️ **Compliance & Security**
- Immutable audit trails
- GDPR/SOC2 compliance reporting
- Data classification and PII handling
- Security event tracking

### 📈 **Automated Reporting**
- Daily/weekly/monthly reports
- Multiple output formats (HTML, JSON, PDF)
- Scheduled report generation
- Custom templates

## 🎯 Supported Agent Types

- **Research Agents** - Search and data collection tracking
- **Report Agents** - Document generation and processing metrics
- **Editor Agents** - Content modification and version control
- **UI Coordinators** - User interaction and session management
- **Custom Agents** - Extensible framework for any agent type

## 🔧 Installation

### Basic Setup
```bash
# Core logging (always works)
pip install asyncio dataclasses datetime typing

# Advanced monitoring (optional)
pip install streamlit pandas numpy plotly
```

### Environment Setup
```bash
export LOG_LEVEL=INFO
export ENABLE_ADVANCED_MONITORING=true
export LOG_RETENTION_DAYS=30
```

## 🚦 Getting Started in 3 Steps

### 1. **Initialize Monitoring**
```python
from multi_agent_research_system.monitoring_integration import MonitoringIntegration

monitoring = MonitoringIntegration(
    session_id="my_app_session",
    enable_advanced_monitoring=True
)
```

### 2. **Register SDK Hooks**
```python
from multi_agent_research_system.agent_logging import HookIntegration

hooks = HookIntegration(session_id="my_app_session")
await client.register_hooks([hooks])
```

### 3. **Start Your Agents**
```python
# All agent activities are now automatically logged!
# Tool usage, messages, errors, performance - everything is tracked.
```

## 📊 Real-time Dashboard

Launch the monitoring dashboard:
```bash
streamlit run multi_agent_research_system/monitoring/real_time_dashboard.py
```

Features:
- Live system health status
- Performance metrics visualization
- Agent activity tracking
- Error monitoring and alerts

## 🔍 Log Search and Analysis

```python
from multi_agent_research_system.log_analysis import LogAggregator, LogSearchEngine

# Aggregate logs from all sources
aggregator = LogAggregator(session_id="my_session")
await aggregator.aggregate_logs()

# Search for specific events
search_engine = LogSearchEngine()
entries = aggregator.get_entries()
search_engine.build_index(entries)

# Find all errors
from multi_agent_research_system.log_analysis import SearchQuery, SearchOperator
error_query = SearchQuery(field="level", operator=SearchOperator.EQUALS, value="ERROR")
results, stats = search_engine.search(entries, error_query)
```

## 📋 Automated Reports

```python
from multi_agent_research_system.log_analysis import ReportGenerator, ReportType
from datetime import datetime, timedelta

# Generate daily performance report
generator = ReportGenerator(session_id="my_session")
report = await generator.generate_report(
    report_type=ReportType.DAILY_SUMMARY,
    period_start=datetime.now() - timedelta(days=1),
    period_end=datetime.now(),
    formats=['HTML', 'JSON']
)
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
cd multi_agent_research_system
python test_log_analysis.py
```

All tests should pass:
- ✅ LogAggregator tests
- ✅ LogSearchEngine tests
- ✅ AnalyticsEngine tests
- ✅ AuditTrailManager tests
- ✅ ReportGenerator tests
- ✅ Integration tests

## 🚨 Production Considerations

### Performance
- Async logging for non-blocking operations
- Configurable retention policies
- Graceful degradation when dependencies unavailable
- Memory-efficient indexing and caching

### Security
- Configurable data classification levels
- PII detection and handling
- Secure log storage with integrity verification
- Access control integration

### Reliability
- Automatic log rotation
- Background cleanup tasks
- Health monitoring and alerting
- Fallback to basic logging if advanced features fail

## 📁 File Structure

```
multi_agent_research_system/
├── agent_logging/              # Core logging framework
│   ├── structured_logger.py   # Structured JSON logging
│   ├── hook_integration.py    # SDK hook integration
│   ├── agent_logger.py       # Agent-specific logging
│   └── agent_loggers/        # Specialized agent loggers
├── monitoring/               # Advanced monitoring features
│   ├── metrics_collector.py  # Real-time metrics
│   ├── performance_monitor.py # Performance tracking
│   ├── system_health.py      # Health monitoring
│   ├── real_time_dashboard.py # Streamlit dashboard
│   └── diagnostics.py        # Diagnostic tools
├── log_analysis/            # Analytics and reporting
│   ├── log_aggregator.py     # Log collection
│   ├── log_search.py        # Search functionality
│   ├── analytics_engine.py  # Analytics and insights
│   ├── audit_trail.py       # Compliance tracking
│   └── report_generator.py  # Report generation
├── monitoring_integration.py # Unified interface
└── test_log_analysis.py     # Comprehensive test suite
```

## 🎯 Use Cases

### **Development Teams**
- Debug multi-agent interactions
- Performance optimization
- Error tracking and resolution

### **Operations Teams**
- System health monitoring
- Capacity planning
- Incident response

### **Compliance Teams**
- Audit trail generation
- Regulatory reporting
- Data governance

### **Business Teams**
- Usage analytics
- Performance insights
- ROI tracking

## 🔧 Configuration Options

### Basic Configuration
```python
monitoring = MonitoringIntegration(
    session_id="unique_session_id",
    monitoring_dir="custom_monitoring_dir",
    enable_advanced_monitoring=True/False
)
```

### Environment Variables
```bash
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_DIR=logs                     # Directory for log files
MONITORING_DIR=monitoring        # Directory for monitoring data
ENABLE_ADVANCED_MONITORING=true  # Enable/disable advanced features
LOG_RETENTION_DAYS=30           # Log retention period
METRICS_COLLECTION_INTERVAL=30   # Metrics collection frequency (seconds)
```

## 🆘 Troubleshooting

### **No logs appearing?**
1. Check log directory permissions
2. Verify session_id consistency
3. Ensure hooks are registered with SDK
4. Check log level configuration

### **Advanced monitoring not working?**
1. Verify optional dependencies are installed
2. Check `MONITORING_AVAILABLE` flag
3. Review environment configuration
4. Test with `enable_advanced_monitoring=False`

### **Performance issues?**
1. Increase log retention cleanup intervals
2. Reduce log verbosity (use INFO instead of DEBUG)
3. Monitor disk space usage
4. Consider disabling advanced features if not needed

## 🤝 Contributing

When adding new features:
1. Follow the existing 5-phase architecture
2. Add comprehensive tests
3. Update documentation
4. Ensure backward compatibility
5. Test graceful degradation scenarios

## 📄 License

This logging system is part of the multi-agent research system and follows the same licensing terms.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting sections in the documentation
2. Run the test suite to verify functionality
3. Review system health and configuration
4. Check log files for error messages

---

**Ready to get started?** 🚀

1. Read the [Implementation Guide](logging_implementation_guide.md) for step-by-step setup
2. Use the [Quick Reference](logging_quick_reference.md) for common tasks
3. Refer to the [Complete Documentation](comprehensive_logging_system.md) for detailed information

**Transform your multi-agent debugging and monitoring experience today!** 🎯