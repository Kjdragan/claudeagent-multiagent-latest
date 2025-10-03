# Setup and Run Guide

## Quick Start

### Option 1: Web Interface (Recommended)
```bash
# 1. Install dependencies
uv add --dev streamlit

# 2. Run the web interface
uv run streamlit run ui/streamlit_app.py

# The app will open in your browser at http://localhost:8501
```

### Option 2: Programmatic Usage
```bash
# 1. Install dependencies (already done with uv)
uv run python example_usage.py

# 2. Run the system test
uv run python test_system.py
```

### Option 3: Full Testing Suite
```bash
# Run basic tests (no API key needed)
uv run pytest tests/unit/ -v

# Run all tests (requires API key)
export ANTHROPIC_API_KEY="your-api-key"
python tests/run_tests.py quick
```

## Requirements

### For Basic Operation (No API Key)
- ✅ Python 3.8+
- ✅ UV package manager
- ✅ Basic dependencies (already installed)

### For Full Functionality (With Real API)
- 📝 Anthropic API key (`ANTHROPIC_API_KEY`)
- 📝 Internet connection for web searches
- 📝 Additional dependencies

## Getting API Key

1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Create an account or sign in
3. Generate an API key
4. Set environment variable:
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```

## Directory Structure

```
multi_agent_research_system/
├── ui/streamlit_app.py        # Web interface
├── example_usage.py           # Programmatic example
├── test_system.py            # System validation
├── core/                     # Core system components
├── config/                   # Agent configurations
├── tests/                    # Test suite
└── researchmaterials/        # Output directory (auto-created)
```

## Running Options

### 1. Web Interface (Streamlit)
```bash
cd multi_agent_research_system
uv run streamlit run ui/streamlit_app.py
```

**Features:**
- 📝 Form-based research requests
- 📊 Real-time progress monitoring
- 📄 Report viewing and download
- 🔍 Session management

### 2. Command Line Example
```bash
cd multi_agent_research_system
uv run python example_usage.py
```

**Features:**
- 🔬 Demonstrates programmatic usage
- 📊 Shows workflow progression
- 🎯 Example research topic
- 💡 API usage guidance

### 3. System Validation
```bash
cd multi_agent_research_system
uv run python test_system.py
```

**Features:**
- ✅ System health check
- 🔍 Component validation
- 📋 Readiness verification
- 💡 Setup guidance

### 4. Development Testing
```bash
cd multi_agent_research_system

# Quick test
python tests/run_tests.py quick

# Full test suite (with API key)
export ANTHROPIC_API_KEY="your-key"
python tests/run_tests.py all
```

## What to Expect

### Without API Key (Mock Mode)
- ✅ All UI features work
- ✅ Session management works
- ✅ File operations work
- 🔍 Mock research results
- 📝 Simulated workflow progression

### With API Key (Full Mode)
- ✅ Real research with web searches
- ✅ Actual Claude AI responses
- ✅ Quality report generation
- 🔍 Real findings and sources
- 📊 Professional research outputs

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Make sure you're in the right directory
cd /home/kjdragan/lrepos/claude-agent-sdk-python/multi_agent_research_system

# Check dependencies
uv sync
```

**Streamlit Not Found:**
```bash
# Install streamlit
uv add --dev streamlit

# Or use pip
uv run pip install streamlit
```

**Permission Issues:**
```bash
# Make sure researchmaterials directory can be created
mkdir -p researchmaterials/sessions
chmod 755 researchmaterials
```

**API Key Issues:**
```bash
# Check if API key is set
echo $ANTHROPIC_API_KEY

# Set temporarily for current session
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Getting Help

1. **Check the logs** - Both Streamlit and Python show detailed error messages
2. **Run system test** - `uv run python test_system.py` validates basic setup
3. **Check dependencies** - `uv sync` ensures all packages are installed
4. **Verify API key** - Make sure `ANTHROPIC_API_KEY` is set correctly

## Example Usage

### Web Interface Workflow
1. Open http://localhost:8501
2. Fill in research request form:
   - Topic: "The Impact of AI on Healthcare"
   - Depth: "Standard Research"
   - Audience: "Technical"
   - Format: "Technical Documentation"
3. Click "Start Research"
4. Monitor progress in real-time
5. Download final report

### Programmatic Workflow
```python
from multi_agent_research_system import ResearchOrchestrator

# Initialize the system
orchestrator = ResearchOrchestrator()
await orchestrator.initialize()

# Start research
session_id = await orchestrator.start_research_session(
    topic="AI in Healthcare",
    user_requirements={
        "depth": "Standard Research",
        "audience": "Technical",
        "format": "Technical Documentation"
    }
)

# Monitor progress
status = await orchestrator.get_session_status(session_id)
print(f"Status: {status['status']}")
```

## Next Steps

1. **Try the web interface** - Most user-friendly option
2. **Run the example** - See how it works programmatically
3. **Set up API key** - For real research functionality
4. **Explore tests** - Understand the system capabilities
5. **Customize agents** - Modify agent prompts and tools as needed

Enjoy using the multi-agent research system! 🚀