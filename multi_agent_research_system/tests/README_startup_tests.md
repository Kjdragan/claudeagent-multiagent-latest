# Startup Health Tests Documentation

## Overview

The startup health tests provide comprehensive validation of the multi-agent research system before running actual research. These tests were extracted from the main orchestrator to streamline normal operation while preserving debugging capabilities.

## Purpose

### Why Were These Tests Created?

Previously, the multi-agent research system performed health checks during every startup, which:
- Added unnecessary delay to research sessions
- Consumed API resources on test searches
- Generated noise in production logs
- Made debugging more difficult

These tests are now available separately for development, troubleshooting, and validation purposes.

### What Do These Tests Validate?

#### 1. Agent Health Checks (`check_agent_health`)
- **Connectivity**: Verifies each agent can be reached via the Claude SDK
- **Response Generation**: Confirms agents can produce substantive responses
- **Configuration**: Validates agent definitions and prompts are working
- **Performance**: Measures response times for each agent

#### 2. Tool Verification (`verify_tool_execution`)
- **SERP API Search**: Tests web search functionality via MCP server
- **File Operations**: Validates Read/Write tool availability
- **MCP Integration**: Confirms proper tool routing and permissions
- **Tool Access**: Ensures agents can access required tools

## Usage

### Running All Tests

```bash
# Basic test run
python tests/test_startup_health.py

# With verbose output
python tests/test_startup_health.py --verbose

# Save results to file
python tests/test_startup_health.py --output health_report.json
```

### Running Specific Tests

```bash
# Test only agents
python tests/test_startup_health.py --test agents

# Test only tools
python tests/test_startup_health.py --test tools

# Tool test with verbose output
python tests/test_startup_health.py --test tools --verbose
```

### Integration with Research CLI

```bash
# Run health checks before research
python run_research.py "your topic" --verify-startup

# Skip all health checks for fastest startup
python run_research.py "your topic" --quick-start

# Normal operation (no health checks)
python run_research.py "your topic"
```

## Understanding Test Results

### Healthy System Output

```
🏥 MULTI-AGENT RESEARCH SYSTEM - STARTUP HEALTH REPORT
========================================================
Timestamp: 2025-10-04T01:55:00.000000
Overall Status: HEALTHY

------------------------------------------------------------
AGENT HEALTH
------------------------------------------------------------
Total Agents: 4
Healthy: 4
Unhealthy: 0
   ✅ research_agent: healthy (2.34s)
   ✅ report_agent: healthy (1.87s)
   ✅ editor_agent: healthy (2.12s)
   ✅ ui_coordinator: healthy (1.98s)

------------------------------------------------------------
TOOL VERIFICATION
------------------------------------------------------------
Agent Tested: research_agent
Tools Working: 2/2
Success Rate: 100.0%
   ✅ SERP API Search: Working
   ✅ File Operations: Working
```

### Common Issues and Solutions

#### Agent Health Issues

**Symptom:** `❌ research_agent: No substantive response`
- **Cause**: Agent configuration or SDK connectivity issue
- **Solution**: Check ANTHROPIC_API_KEY and agent definitions

**Symptom:** `❌ report_agent: Agent test failed: timeout`
- **Cause**: Network connectivity or API rate limiting
- **Solution**: Check network connection and API quotas

#### Tool Verification Issues

**Symptom:** `❌ SERP API Search: Tool not executed in response`
- **Cause**: MCP server not running or SERP_API_KEY missing
- **Solution**:
  - Set SERP_API_KEY environment variable
  - Check MCP server initialization

**Symptom:** `❌ File Operations: Failed: Permission denied`
- **Cause**: File system permissions issue
- **Solution**: Check write permissions in working directory

## When to Run These Tests

### Development Workflow
1. **Before Code Changes**: Establish baseline health
2. **After Agent Changes**: Validate configuration updates
3. **Before Deployments**: Ensure system readiness
4. **Troubleshooting**: Diagnose reported issues

### Environment Validation
- **New Setup**: Verify all components are working
- **After Updates**: Confirm system stability
- **Migration Testing**: Validate new environments

### Production Monitoring
- **Scheduled Health Checks**: Automated validation
- **Incident Response**: Quick system validation
- **Performance Baselines**: Track system performance over time

## Test Architecture

### Components

```
tests/test_startup_health.py
├── StartupHealthTester (main class)
├── check_agent_health() - Agent connectivity testing
├── verify_tool_execution() - Tool functionality testing
├── run_all_tests() - Comprehensive test suite
└── print_report() - Formatted result display
```

### Dependencies

- **Claude Agent SDK**: For agent communication
- **Research Orchestrator**: For system initialization
- **MCP Server**: For tool testing
- **SERP API**: For search validation

### Test Data

- **Test Queries**: Simple, non-sensitive queries like "test query"
- **File Operations**: Non-existent file tests (test.txt)
- **Agent Prompts**: Capability introduction requests

## Performance Considerations

### Test Duration
- **Agent Health**: ~8-12 seconds (4 agents × 2-3 seconds each)
- **Tool Verification**: ~5-10 seconds (2 tools)
- **Total Runtime**: 15-25 seconds typically

### Resource Usage
- **API Calls**: Minimal test queries (costs < $0.01)
- **Network**: SERP API calls for search testing
- **Memory**: Standard orchestrator initialization

### Optimization Tips
- Use `--test agents` for faster agent-only validation
- Use `--test tools` for quick tool checking
- Run tests during development, not production
- Cache successful results during development sessions

## Troubleshooting

### Common Error Messages

#### "ImportError: No module named 'claude_agent_sdk'"
```bash
# Solution: Install the SDK
pip install claude-agent-sdk
```

#### "Failed to initialize orchestrator"
- Check Python path and working directory
- Verify environment variables are set
- Check for missing dependencies

#### "Agent test failed: timeout"
- Check network connectivity
- Verify API key validity
- Check for rate limiting

#### "SERP_API_KEY Status: NOT_SET"
```bash
# Solution: Set the environment variable
export SERP_API_KEY="your_api_key_here"
```

### Debug Mode

For detailed troubleshooting:

```bash
# Enable verbose logging
python tests/test_startup_health.py --verbose

# Or use the debug flag in research CLI
python run_research.py "topic" --verify-startup --debug-agents --log-level DEBUG
```

### Log Analysis

Test logs are saved to `KEVIN/logs/`:
- `startup_health_test.log` - Main test execution log
- Agent-specific logs for detailed debugging

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Run Startup Health Tests
  run: |
    python tests/test_startup_health.py --output health_report.json

- name: Upload Health Report
  uses: actions/upload-artifact@v3
  with:
    name: health-report
    path: health_report.json
```

### Docker Integration

```dockerfile
# In Dockerfile
COPY . /app
WORKDIR /app
RUN python tests/test_startup_health.py --test agents

# Health check for running container
HEALTHCHECK --interval=30s --timeout=10s \
  CMD python tests/test_startup_health.py --test agents || exit 1
```

## Best Practices

### Development
1. **Run Early**: Test before making significant changes
2. **Run Often**: Validate after each major change
3. **Document Results**: Track health over time
4. **Use Flags**: Leverage `--test agents` for faster feedback

### Production
1. **Monitor Regularly**: Schedule automated health checks
2. **Alert on Failures**: Set up notifications for test failures
3. **Maintain Baselines**: Track expected performance metrics
4. **Plan Rollbacks**: Have quick recovery procedures

### Performance
1. **Cache Results**: During development sessions
2. **Parallel Testing**: Run agent and tool tests concurrently when possible
3. **Selective Testing**: Use specific test flags for targeted validation
4. **Resource Management**: Monitor API usage during testing

## FAQ

**Q: Why were these tests removed from the main startup?**
A: To improve startup performance and reduce unnecessary API usage in production.

**Q: Should I run these tests every time?**
A: No, only when troubleshooting, after changes, or during development setup.

**Q: What if tests fail but research still works?**
A: Tests may detect issues that don't immediately break research. Address warnings to prevent future problems.

**Q: Can I customize the test queries?**
A: Yes, modify the test prompts in `test_startup_health.py` for your specific use case.

**Q: How much do these tests cost?**
A: Minimal - usually less than $0.01 per full run due to small test queries.

---

**Last Updated**: October 4, 2025
**Version**: 1.0
**Maintained by**: Multi-Agent Research System Team