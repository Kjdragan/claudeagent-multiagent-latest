"""Streamlit user interface for the multi-agent research system.

This provides a web-based interface for users to initiate and monitor
research projects using the multi-agent system.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.warning("Warning: python-dotenv not found. Using environment variables only.")

# Set up API configuration
ANTHROPIC_BASE_URL = os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

if ANTHROPIC_API_KEY:
    # Set environment variables for the SDK
    os.environ['ANTHROPIC_BASE_URL'] = ANTHROPIC_BASE_URL
    os.environ['ANTHROPIC_API_KEY'] = ANTHROPIC_API_KEY
    st.success(f"✅ Connected to Anthropic API: {ANTHROPIC_BASE_URL}")
else:
    st.error("⚠️  No ANTHROPIC_API_KEY found in environment or .env file")
    st.info("Please set up your .env file with the API key to use real research functionality.")

# Import core modules with proper path handling
try:
    from core.logging_config import (
        get_log_level,
        get_log_summary,
        get_logger,
        set_log_level,
        setup_logging,
    )
    from core.orchestrator import ResearchOrchestrator
except ImportError:
    # Fallback for when running as module
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.logging_config import (
        get_log_level,
        get_log_summary,
        get_logger,
        set_log_level,
        setup_logging,
    )
    from core.orchestrator import ResearchOrchestrator


class ResearchUI:
    """Streamlit UI for the multi-agent research system."""

    def __init__(self):
        # Initialize logging for UI
        setup_logging()
        self.logger = get_logger("streamlit_ui")
        self.logger.info("Streamlit UI initialized")

        # Initialize orchestrator in session state to persist across reruns
        if 'orchestrator' not in st.session_state:
            st.session_state.orchestrator = ResearchOrchestrator(debug_mode=True)
            self.logger.info("Created new orchestrator in session state")
        else:
            self.logger.debug("Using existing orchestrator from session state")

        self.orchestrator = st.session_state.orchestrator
        self.current_session = None

    async def initialize_system(self):
        """Initialize the research orchestrator."""
        # Check if orchestrator is properly initialized
        needs_init = (
            not hasattr(self.orchestrator, 'agent_clients') or
            not self.orchestrator.agent_clients or
            len(self.orchestrator.agent_clients) == 0
        )

        if needs_init:
            with st.spinner("Initializing research system..."):
                self.logger.info("Initializing research orchestrator from UI")
                try:
                    await self.orchestrator.initialize()
                    self.logger.info(f"Research orchestrator initialized successfully with {len(self.orchestrator.agent_clients)} agents")
                    self.logger.debug(f"Available agents: {list(self.orchestrator.agent_clients.keys())}")
                    st.success("Research system initialized successfully!")
                except Exception as e:
                    self.logger.error(f"Failed to initialize orchestrator: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    st.error(f"Failed to initialize research system: {e}")
        else:
            self.logger.debug("Research orchestrator already initialized")

    def run(self):
        """Run the Streamlit application."""
        st.set_page_config(
            page_title="Multi-Agent Research System",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("🔬 Multi-Agent Research System")
        st.markdown("---")

        # Initialize the system
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False

        if not st.session_state.system_initialized:
            asyncio.run(self.initialize_system())
            st.session_state.system_initialized = True

        # Create sidebar for navigation
        self.create_sidebar()

        # Main content area
        if st.session_state.get('current_view') == 'new_research':
            self.show_new_research_form()
        elif st.session_state.get('current_view') == 'session_status':
            self.show_session_status()
        elif st.session_state.get('current_view') == 'results':
            self.show_results()
        elif st.session_state.get('current_view') == 'logs':
            self.show_logs()
        elif st.session_state.get('current_view') == 'debug':
            self.show_debug_info()
        elif st.session_state.get('current_view') == 'kevin':
            self.show_kevin_directory()
        elif st.session_state.get('current_view') == 'live_logs':
            self.show_live_logs()
        else:
            self.show_welcome_page()

    def create_sidebar(self):
        """Create the navigation sidebar."""
        st.sidebar.title("Navigation")

        if st.sidebar.button("🏠 Home", use_container_width=True):
            st.session_state.current_view = 'home'

        if st.sidebar.button("🆕 New Research", use_container_width=True):
            st.session_state.current_view = 'new_research'

        if st.sidebar.button("📊 Session Status", use_container_width=True):
            st.session_state.current_view = 'session_status'

        if st.session_state.get('current_session'):
            if st.sidebar.button("📄 Results", use_container_width=True):
                st.session_state.current_view = 'results'

        if st.sidebar.button("📋 Logs", use_container_width=True):
            st.session_state.current_view = 'logs'

        if st.sidebar.button("🐛 Agent Debug", use_container_width=True):
            st.session_state.current_view = 'debug'

        if st.sidebar.button("📁 KEVIN Directory", use_container_width=True):
            st.session_state.current_view = 'kevin'

        if st.sidebar.button("🔴 Live Logs", use_container_width=True):
            st.session_state.current_view = 'live_logs'

        st.sidebar.markdown("---")

        # System controls
        st.sidebar.subheader("🔧 System Controls")
        if st.sidebar.button("🔄 Reinitialize System", type="secondary"):
            st.session_state.system_initialized = False
            st.sidebar.info("System will be reinitialized on next interaction")
            st.rerun()

        st.sidebar.markdown("---")

        # Log level control
        st.sidebar.subheader("📋 Logging Controls")
        current_log_level = get_log_level()
        selected_log_level = st.sidebar.selectbox(
            "Log Level",
            options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            index=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].index(current_log_level),
            help="Set the logging level for the system"
        )

        if selected_log_level != current_log_level:
            set_log_level(selected_log_level)
            self.logger.info(f"Log level changed to: {selected_log_level}")
            st.sidebar.success(f"Log level set to {selected_log_level}")
            st.rerun()

        # Show log summary
        log_summary = get_log_summary()
        if log_summary:
            with st.sidebar.expander("📊 Log Information"):
                st.write(f"**Current Level:** {log_summary.get('log_level', 'Unknown')}")
                st.write(f"**Log Directory:** {log_summary.get('log_directory', 'Unknown')}")
                if log_summary.get('current_log_file'):
                    current_file = Path(log_summary['current_log_file'])
                    st.write(f"**Current Log File:** {current_file.name}")
                    if current_file.exists():
                        file_size = current_file.stat().st_size
                        st.write(f"**File Size:** {file_size:,} bytes")
                st.write(f"**Files Cleaned on Startup:** {log_summary.get('log_files_deleted', 0)}")

        st.sidebar.markdown("---")

        # Show active sessions
        if hasattr(self.orchestrator, 'active_sessions') and self.orchestrator.active_sessions:
            st.sidebar.subheader("Active Sessions")
            for session_id, session_data in self.orchestrator.active_sessions.items():
                if session_data.get('status') != 'completed':
                    topic = session_data.get('topic', 'Unknown Topic')
                    status = session_data.get('status', 'Unknown')
                    if st.sidebar.button(f"📋 {topic[:30]}...", key=f"session_{session_id}", use_container_width=True):
                        st.session_state.current_session = session_id
                        st.session_state.current_view = 'session_status'

    def show_welcome_page(self):
        """Show the welcome page."""
        st.markdown("## Welcome to the Multi-Agent Research System 🔬")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 What This System Does")
            st.markdown("""
            This system uses multiple specialized AI agents to:

            - **🔍 Conduct comprehensive research** on any topic
            - **📝 Generate well-structured reports** from findings
            - **✅ Review and edit** for quality and accuracy
            - **🔄 Iterate and improve** based on feedback

            Each agent specializes in specific tasks, working together to deliver professional-grade research reports.
            """)

        with col2:
            st.markdown("### 🤖 Meet the Agents")
            st.markdown("""
            **Research Agent** - Expert at finding and analyzing information

            **Report Agent** - Creates structured, professional reports

            **Editor Agent** - Reviews and improves report quality

            **Coordinator** - Manages the entire workflow
            """)

        st.markdown("---")

        st.markdown("### 🚀 Get Started")
        st.markdown("Click 'New Research' in the sidebar to begin your research project.")

        # Show recent completed sessions if any
        if hasattr(self.orchestrator, 'active_sessions'):
            completed_sessions = [
                (sid, data) for sid, data in self.orchestrator.active_sessions.items()
                if data.get('status') == 'completed'
            ]

            if completed_sessions:
                st.markdown("### 📚 Recent Research Projects")
                for session_id, session_data in completed_sessions[-3:]:  # Show last 3
                    topic = session_data.get('topic', 'Unknown Topic')
                    completed_at = session_data.get('completed_at', 'Unknown')
                    st.markdown(f"**{topic}** - Completed {completed_at}")

    def show_new_research_form(self):
        """Show the new research request form."""
        st.markdown("## 🆕 Start New Research")

        with st.form("research_form"):
            st.markdown("### 📋 Research Requirements")

            # Basic information
            topic = st.text_input(
                "🎯 Research Topic",
                placeholder="What would you like to research?",
                help="Be specific about the topic you want to research"
            )

            # Research depth
            depth = st.selectbox(
                "🔍 Research Depth",
                options=["Quick Overview", "Standard Research", "Comprehensive Analysis"],
                index=1,
                help="How thorough should the research be?"
            )

            # Target audience
            audience = st.selectbox(
                "👥 Target Audience",
                options=["General Public", "Academic", "Business", "Technical", "Policy Makers"],
                help="Who is this report for?"
            )

            # Report format
            format_type = st.selectbox(
                "📄 Report Format",
                options=["Standard Report", "Academic Paper", "Business Brief", "Technical Documentation"],
                help="What format should the final report be in?"
            )

            # Specific requirements
            requirements = st.text_area(
                "📝 Specific Requirements",
                placeholder="Any specific aspects you want the research to focus on? Questions that need answering? Specific sources to include?",
                help="Tell us about any specific requirements or questions you need addressed"
            )

            # Timeline
            timeline = st.selectbox(
                "⏰ Timeline",
                options=["ASAP", "Within 24 hours", "Within 3 days", "Within 1 week", "Flexible"],
                index=0
            )

            # Submit button
            submit_button = st.form_submit_button("🚀 Start Research", type="primary")

            if submit_button:
                if not topic.strip():
                    st.error("Please provide a research topic.")
                else:
                    self.start_research_session({
                        'topic': topic.strip(),
                        'depth': depth,
                        'audience': audience,
                        'format': format_type,
                        'requirements': requirements,
                        'timeline': timeline
                    })

    def start_research_session(self, requirements: dict[str, Any]):
        """Start a new research session."""
        try:
            # Ensure orchestrator is initialized
            if not hasattr(self.orchestrator, 'agent_clients') or not self.orchestrator.agent_clients:
                self.logger.warning("Orchestrator not initialized, attempting to initialize...")
                asyncio.run(self.initialize_system())

            # Check if research_agent is available
            if "research_agent" not in self.orchestrator.agent_clients:
                self.logger.error(f"research_agent not found in agent_clients: {list(self.orchestrator.agent_clients.keys())}")
                st.error("Research agent not available. Please reinitialize the system.")
                return

            with st.spinner("Starting research session..."):
                self.logger.info(f"Starting research session for topic: {requirements['topic']}")

                # Start the research session in background without blocking
                session_id = self.orchestrator._create_session_id()

                # Create session directory in KEVIN structure
                from pathlib import Path
                session_path = Path(f"KEVIN/sessions/{session_id}")
                session_path.mkdir(parents=True, exist_ok=True)
                # Create subdirectories for organization
                (session_path / "research").mkdir(exist_ok=True)
                (session_path / "working").mkdir(exist_ok=True)
                (session_path / "final").mkdir(exist_ok=True)

                self.orchestrator.active_sessions[session_id] = {
                    "session_id": session_id,
                    "topic": requirements['topic'],
                    "user_requirements": requirements,
                    "status": "starting",
                    "created_at": datetime.now().isoformat(),
                    "current_stage": "initialization",
                    "workflow_history": [],
                    "final_report": None
                }

                # Save session state
                asyncio.run(self.orchestrator.save_session_state(session_id))

                # Start the research workflow in background
                import threading
                def run_research_background():
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(self.orchestrator.execute_research_workflow(session_id))
                    except Exception as e:
                        self.logger.error(f"Background research error: {e}")
                    finally:
                        loop.close()

                # Start background thread
                thread = threading.Thread(target=run_research_background, daemon=True)
                thread.start()

                st.session_state.current_session = session_id
                st.session_state.current_view = 'session_status'

                st.success(f"Research session started! Session ID: {session_id}")
                st.rerun()

        except Exception as e:
            self.logger.error(f"Failed to start research session: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            st.error(f"Failed to start research session: {e}")

    def show_session_status(self):
        """Show the status of the current research session."""
        st.markdown("## 📊 Research Session Status")

        session_id = st.session_state.get('current_session')

        if not session_id:
            st.warning("No active research session. Please start a new research project.")
            return

        # Get session data
        session_data = self.orchestrator.active_sessions.get(session_id)
        if not session_data:
            st.error("Session not found.")
            return

        # Session overview
        st.markdown("### 📋 Session Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Topic", session_data.get('topic', 'Unknown')[:20] + '...')

        with col2:
            status = session_data.get('status', 'Unknown')
            status_color = {
                'researching': '🔍',
                'generating_report': '📝',
                'editorial_review': '✅',
                'finalizing': '🎯',
                'completed': '✅',
                'error': '❌'
            }.get(status, '📋')
            st.metric("Status", f"{status_color} {status.replace('_', ' ').title()}")

        with col3:
            stage = session_data.get('current_stage', 'Unknown')
            st.metric("Current Stage", stage.replace('_', ' ').title())

        with col4:
            created_at = session_data.get('created_at', 'Unknown')
            st.metric("Started", created_at[:10] if created_at != 'Unknown' else 'Unknown')

        # Progress indicator
        self.show_progress_indicator(session_data)

        # Workflow history
        st.markdown("### 📈 Workflow Progress")
        workflow_history = session_data.get('workflow_history', [])

        if workflow_history:
            for i, stage in enumerate(workflow_history):
                stage_name = stage.get('stage', 'Unknown Stage')
                completed_at = stage.get('completed_at', 'Unknown')
                results_count = stage.get('results_count', 0)

                with st.expander(f"🔄 {stage_name.replace('_', ' ').title()}"):
                    st.markdown(f"**Completed:** {completed_at}")
                    st.markdown(f"**Results Generated:** {results_count} items")

                    # Show status-specific information
                    if stage_name == 'research' and 'research_results' in session_data:
                        st.markdown("**Research Materials:** Collected and analyzed")
                    elif stage_name == 'report_generation' and 'report_results' in session_data:
                        st.markdown("**Report:** Generated and saved")
                    elif stage_name == 'editorial_review' and 'review_results' in session_data:
                        st.markdown("**Editorial Review:** Completed with feedback")
                    elif stage_name == 'revisions' and 'revision_results' in session_data:
                        st.markdown("**Revisions:** Applied based on feedback")
        else:
            st.info("Workflow in progress...")

        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Refresh Status", type="secondary"):
                st.rerun()

        with col2:
            if session_data.get('status') == 'completed':
                if st.button("📄 View Results", type="primary"):
                    st.session_state.current_view = 'results'
                    st.rerun()

        with col3:
            if st.button("❌ Cancel Session", type="secondary"):
                self.cancel_session(session_id)

    def show_progress_indicator(self, session_data: dict[str, Any]):
        """Show a progress indicator for the research workflow."""
        stages = ['research', 'report_generation', 'editorial_review', 'finalization']
        current_stage = session_data.get('current_stage', 'research')
        status = session_data.get('status', 'researching')

        if status == 'error':
            st.error("⚠️ Research session encountered an error")
            return

        # Calculate progress
        progress_value = 0
        stage_labels = []

        for i, stage in enumerate(stages):
            stage_label = stage.replace('_', ' ').title()
            stage_labels.append(stage_label)

            if stage == current_stage:
                # Current stage - partial progress
                progress_value = i + 0.5
                break
            elif stage in [h.get('stage') for h in session_data.get('workflow_history', [])]:
                # Completed stage
                progress_value = i + 1
            else:
                # Not reached yet
                break

        # Show progress bar
        st.markdown("### 📊 Progress")
        st.progress(progress_value / len(stages))

        # Show stage labels
        cols = st.columns(len(stages))
        for i, (col, label) in enumerate(zip(cols, stage_labels, strict=False)):
            with col:
                if i < progress_value:
                    st.markdown(f"✅ {label}")
                elif i == int(progress_value):
                    st.markdown(f"🔄 {label}")
                else:
                    st.markdown(f"⏳ {label}")

    def show_results(self):
        """Show the research results."""
        st.markdown("## 📄 Research Results")

        session_id = st.session_state.get('current_session')
        if not session_id:
            st.warning("No session selected.")
            return

        session_data = self.orchestrator.active_sessions.get(session_id)
        if not session_data:
            st.error("Session not found.")
            return

        if session_data.get('status') != 'completed':
            st.warning("Research is not yet completed. Please check back later.")
            return

        st.success(f"✅ Research completed on {session_data.get('completed_at', 'Unknown')}")

        # Show report download
        st.markdown("### 📥 Download Report")

        # Look for generated report files
        from pathlib import Path

        session_path = Path(f"KEVIN/sessions/{session_id}")
        if session_path.exists():
            # Look for reports in the final subdirectory
            final_dir = session_path / "final"
            if final_dir.exists():
                report_files = list(final_dir.glob("*.md"))
            else:
                report_files = list(session_path.glob("*.md"))  # Fallback for old structure

            if report_files:
                for report_file in report_files:
                    with open(report_file, encoding='utf-8') as f:
                        report_content = f.read()

                    # Create download button
                    st.download_button(
                        label=f"📄 Download {report_file.name}",
                        data=report_content,
                        file_name=report_file.name,
                        mime="text/markdown"
                    )

                # Show report preview
                st.markdown("### 👀 Report Preview")
                with st.expander("View Full Report"):
                    st.markdown(report_content)
            else:
                st.warning("No report files found.")
        else:
            st.warning("Research session directory not found.")

        # Action buttons
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🏠 Start New Research", type="primary"):
                st.session_state.current_view = 'new_research'
                st.rerun()

        with col2:
            if st.button("📊 Back to Status", type="secondary"):
                st.session_state.current_view = 'session_status'
                st.rerun()

    def show_logs(self):
        """Show the logs page with log viewing capabilities."""
        st.markdown("## 📋 System Logs")

        # Get log summary
        log_summary = get_log_summary()

        if not log_summary:
            st.error("Logging system not initialized")
            return

        # Show log information
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Current Log Level", log_summary.get('log_level', 'Unknown'))

        with col2:
            current_file = Path(log_summary.get('current_log_file', ''))
            if current_file.exists():
                file_size = current_file.stat().st_size
                st.metric("Log File Size", f"{file_size:,} bytes")
            else:
                st.metric("Log File Size", "No file")

        with col3:
            st.metric("Files Cleaned", log_summary.get('log_files_deleted', 0))

        # Log file controls
        st.markdown("---")
        st.markdown("### 📁 Log File Operations")

        current_log_file = Path(log_summary.get('current_log_file', ''))

        if current_log_file.exists():
            # File info
            st.info(f"**Current Log File:** `{current_log_file.name}`")
            st.info(f"**Full Path:** `{current_log_file}`")

            # Download log file
            with open(current_log_file, encoding='utf-8') as f:
                log_content = f.read()

            st.download_button(
                label="📥 Download Log File",
                data=log_content,
                file_name=current_log_file.name,
                mime="text/plain"
            )

            # Log content viewer
            st.markdown("---")
            st.markdown("### 📖 Log Content Viewer")

            # Line count filter
            col1, col2 = st.columns([1, 3])
            with col1:
                max_lines = st.number_input(
                    "Max lines to display",
                    min_value=10,
                    max_value=10000,
                    value=100,
                    help="Maximum number of log lines to display"
                )
            with col2:
                filter_level = st.selectbox(
                    "Filter by log level",
                    options=["All", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    index=0,
                    help="Show only logs at or above this level"
                )

            # Filter and display logs
            lines = log_content.split('\n')
            if filter_level != "All":
                filtered_lines = [
                    line for line in lines
                    if filter_level in line
                ]
            else:
                filtered_lines = lines

            # Display with pagination
            if filtered_lines:
                total_lines = len(filtered_lines)
                displayed_lines = filtered_lines[-max_lines:]

                st.info(f"Showing {len(displayed_lines)} of {total_lines} lines")

                # Show log content in expandable sections
                log_text = '\n'.join(displayed_lines)
                st.text_area(
                    "Log Content",
                    value=log_text,
                    height=400,
                    help="Log content (read-only)"
                )
            else:
                st.warning("No log entries match the current filter")

        else:
            st.warning("No log file found. The logging system may not have been initialized yet.")

        # Log management
        st.markdown("---")
        st.markdown("### 🛠️ Log Management")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Refresh Logs", type="secondary"):
                self.logger.info("Log refresh requested from UI")
                st.rerun()

        with col2:
            if st.button("🗑️ Clear Log Directory", type="secondary"):
                try:
                    log_dir = Path(log_summary.get('log_directory', 'logs'))
                    if log_dir.exists():
                        import shutil
                        shutil.rmtree(log_dir)
                        log_dir.mkdir(exist_ok=True)
                        self.logger.info("Log directory cleared from UI")
                        st.success("Log directory cleared successfully")
                        st.rerun()
                    else:
                        st.warning("Log directory not found")
                except Exception as e:
                    st.error(f"Error clearing log directory: {e}")

    def show_kevin_directory(self):
        """Show the KEVIN directory contents for debugging."""
        st.markdown("## 📁 KEVIN Directory Contents")

        kevin_dir = Path("/home/kjdragan/lrepos/claude-agent-sdk-python/KEVIN")

        if not kevin_dir.exists():
            st.error("KEVIN directory not found!")
            st.info(f"Expected path: {kevin_dir}")
            return

        # Directory info
        col1, col2, col3 = st.columns(3)
        with col1:
            all_files = list(kevin_dir.rglob("*"))
            file_count = len([f for f in all_files if f.is_file()])
            st.metric("Total Files", file_count)

        with col2:
            web_search_files = list(kevin_dir.glob("web_search_results_*.json"))
            st.metric("Web Search Files", len(web_search_files))

        with col3:
            report_files = list(kevin_dir.glob("research_report_*.md")) + list(kevin_dir.glob("research_report_*.txt"))
            st.metric("Report Files", len(report_files))

        st.markdown("---")

        # Refresh button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh KEVIN Directory", type="secondary"):
                st.rerun()
        with col2:
            if st.button("🗑️ Clear KEVIN Directory", type="secondary"):
                try:
                    import shutil
                    shutil.rmtree(kevin_dir)
                    kevin_dir.mkdir(exist_ok=True)
                    self.logger.info("KEVIN directory cleared from UI")
                    st.success("KEVIN directory cleared successfully")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing KEVIN directory: {e}")

        st.markdown("---")

        # Show web search results
        web_search_files = sorted(kevin_dir.glob("web_search_results_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if web_search_files:
            st.markdown("### 🔍 Web Search Results (Most Recent First)")
            for search_file in web_search_files[:10]:  # Show last 10
                try:
                    with open(search_file, encoding='utf-8') as f:
                        search_data = json.load(f)

                    file_time = datetime.fromtimestamp(search_file.stat().st_mtime)
                    search_query = search_data.get('search_query', 'Unknown Query')
                    result_length = len(search_data.get('search_results', ''))
                    sources_count = len(search_data.get('sources_found', '').split('\n')) if search_data.get('sources_found') else 0

                    with st.expander(f"🔍 {search_query[:50]}... ({file_time.strftime('%H:%M:%S')})"):
                        st.markdown(f"**Search Query:** {search_query}")
                        st.markdown(f"**Result Length:** {result_length} characters")
                        st.markdown(f"**Sources Found:** {sources_count}")
                        st.markdown(f"**File:** `{search_file.name}`")

                        # Show search results
                        search_results = search_data.get('search_results', '')
                        if search_results:
                            st.markdown("**Search Results:**")
                            st.text_area("Content", value=search_results[:1000] + "..." if len(search_results) > 1000 else search_results, height=150, disabled=True)

                        # Show sources
                        sources = search_data.get('sources_found', '')
                        if sources:
                            st.markdown("**Sources:**")
                            st.text_area("Sources", value=sources, height=100, disabled=True)

                        # Download button
                        with open(search_file, encoding='utf-8') as f:
                            st.download_button(
                                label=f"📥 Download {search_file.name}",
                                data=f.read(),
                                file_name=search_file.name,
                                mime="application/json"
                            )
                except Exception as e:
                    st.error(f"Error reading {search_file.name}: {e}")

        # Show research reports
        report_files = sorted(kevin_dir.glob("research_report_*.*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if report_files:
            st.markdown("### 📄 Research Reports (Most Recent First)")
            for report_file in report_files[:5]:  # Show last 5
                try:
                    file_time = datetime.fromtimestamp(report_file.stat().st_mtime)
                    file_size = report_file.stat().st_size

                    with st.expander(f"📄 {report_file.name} ({file_time.strftime('%H:%M:%S')}, {file_size} bytes)"):
                        # Show preview
                        with open(report_file, encoding='utf-8') as f:
                            content = f.read()

                        st.markdown(f"**File:** `{report_file.name}`")
                        st.markdown(f"**Size:** {file_size} characters")

                        # Show preview
                        preview = content[:1000] + "..." if len(content) > 1000 else content
                        st.markdown("**Preview:**")
                        st.text_area("Content", value=preview, height=200, disabled=True)

                        # Download button
                        st.download_button(
                            label=f"📥 Download {report_file.name}",
                            data=content,
                            file_name=report_file.name,
                            mime="text/markdown" if report_file.suffix == '.md' else "text/plain"
                        )
                except Exception as e:
                    st.error(f"Error reading {report_file.name}: {e}")

        # Show other files
        other_files = [f for f in kevin_dir.iterdir() if f.is_file() and not f.name.startswith(('web_search_results_', 'research_report_'))]
        if other_files:
            st.markdown("### 📋 Other Files")
            for file_path in sorted(other_files, key=lambda x: x.stat().st_mtime, reverse=True):
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                file_size = file_path.stat().st_size
                st.markdown(f"📋 `{file_path.name}` - {file_time.strftime('%Y-%m-%d %H:%M:%S')} ({file_size} bytes)")

    def show_live_logs(self):
        """Show live logs with auto-refresh."""
        st.markdown("## 🔴 Live System Logs")

        # Auto-refresh control
        auto_refresh = st.checkbox("🔄 Auto-refresh", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", min_value=1, max_value=10, value=2)

        if auto_refresh:
            st.rerun()

        # Get latest log file
        log_summary = get_log_summary()
        if log_summary and log_summary.get('current_log_file'):
            current_log_file = Path(log_summary['current_log_file'])

            if current_log_file.exists():
                st.info(f"**Watching:** `{current_log_file.name}`")

                # Read last N lines
                lines_to_show = st.number_input("Lines to show", min_value=10, max_value=1000, value=50)

                with open(current_log_file, encoding='utf-8') as f:
                    all_lines = f.readlines()

                # Get last N lines
                recent_lines = all_lines[-lines_to_show:]

                # Filter for interesting lines
                filter_keywords = st.multiselect(
                    "Filter by keywords",
                    options=["🔥", "ERROR", "WARNING", "session", "agent", "tool", "research"],
                    default=["🔥", "ERROR", "WARNING"]
                )

                if filter_keywords:
                    filtered_lines = [
                        line for line in recent_lines
                        if any(keyword in line for keyword in filter_keywords)
                    ]
                else:
                    filtered_lines = recent_lines

                # Show logs
                if filtered_lines:
                    st.markdown(f"**Showing {len(filtered_lines)} lines**")

                    for i, line in enumerate(filtered_lines):
                        # Color coding
                        if "🔥" in line:
                            st.markdown(f'<span style="color: red; font-weight: bold;">{line.strip()}</span>', unsafe_allow_html=True)
                        elif "ERROR" in line:
                            st.markdown(f'<span style="color: red;">{line.strip()}</span>', unsafe_allow_html=True)
                        elif "WARNING" in line:
                            st.markdown(f'<span style="color: orange;">{line.strip()}</span>', unsafe_allow_html=True)
                        elif "session" in line:
                            st.markdown(f'<span style="color: blue;">{line.strip()}</span>', unsafe_allow_html=True)
                        else:
                            st.text(line.strip())
                else:
                    st.warning("No lines match the current filters")

                # Manual refresh button
                if st.button("🔄 Refresh Now"):
                    st.rerun()

            else:
                st.error("Log file not found")
        else:
            st.error("No log files available")

    def cancel_session(self, session_id: str):
        """Cancel a research session."""
        if session_id in self.orchestrator.active_sessions:
            self.orchestrator.active_sessions[session_id]['status'] = 'cancelled'
            st.success("Research session cancelled.")
            st.rerun()

    def show_debug_info(self):
        """Show agent debugging information."""
        st.markdown("## 🐛 Agent Debug Information")

        # Check if orchestrator has debug capabilities
        if not hasattr(self.orchestrator, 'get_debug_output'):
            st.warning("Debug capabilities not available in current orchestrator instance")
            return

        # Get debug output
        debug_output = self.orchestrator.get_debug_output()

        st.markdown("### 📊 Debug Output Summary")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Debug Lines", len(debug_output))

        with col2:
            error_lines = [line for line in debug_output if "ERROR" in line.upper()]
            st.metric("Error Messages", len(error_lines))

        with col3:
            tool_lines = [line for line in debug_output if "tool" in line.lower()]
            st.metric("Tool Operations", len(tool_lines))

        st.markdown("---")

        # Debug output controls
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Refresh Debug Info", type="secondary"):
                self.logger.info("Debug info refresh requested")
                st.rerun()

        with col2:
            if st.button("🗑️ Clear Debug Buffer", type="secondary"):
                self.orchestrator.clear_debug_output()
                self.logger.info("Debug buffer cleared")
                st.success("Debug buffer cleared")
                st.rerun()

        st.markdown("---")

        # Filter options
        st.markdown("### 🔍 Filter Debug Output")
        filter_keyword = st.text_input("Filter by keyword:", placeholder="e.g., ERROR, tool, response...")

        # Apply filter
        filtered_output = debug_output
        if filter_keyword:
            filtered_output = [line for line in debug_output if filter_keyword.lower() in line.lower()]

        st.markdown(f"Showing {len(filtered_output)} of {len(debug_output)} debug lines")

        # Show debug output
        st.markdown("### 📋 Debug Output")
        if filtered_output:
            # Show in expandable sections
            lines_per_section = 50
            for i in range(0, len(filtered_output), lines_per_section):
                section_lines = filtered_output[i:i + lines_per_section]
                start_line = i + 1
                end_line = min(i + lines_per_section, len(filtered_output))

                with st.expander(f"Lines {start_line}-{end_line}"):
                    for line in section_lines:
                        # Color code based on content
                        if "ERROR" in line.upper():
                            st.markdown(f'<span style="color: red;">{line}</span>', unsafe_allow_html=True)
                        elif "WARNING" in line.upper():
                            st.markdown(f'<span style="color: orange;">{line}</span>', unsafe_allow_html=True)
                        elif "tool" in line.lower():
                            st.markdown(f'<span style="color: blue;">{line}</span>', unsafe_allow_html=True)
                        else:
                            st.text(line)

            # Download option
            debug_text = "\n".join(filtered_output)
            st.download_button(
                label="💾 Download Debug Output",
                data=debug_text,
                file_name=f"agent_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        else:
            st.info("No debug output available. Run a research session to see agent debugging information.")

        # Agent status information
        st.markdown("---")
        st.markdown("### 🤖 Agent Status")

        if hasattr(self.orchestrator, 'agent_clients'):
            agent_info = []
            for agent_name, client in self.orchestrator.agent_clients.items():
                try:
                    # Get agent options if available
                    if hasattr(client, 'options'):
                        agent_info.append({
                            "name": agent_name,
                            "model": getattr(client.options, 'model', 'unknown'),
                            "tools": len(getattr(client.options, 'allowed_tools', [])),
                            "debug_hooks": bool(getattr(client.options, 'hooks', None))
                        })
                    else:
                        agent_info.append({
                            "name": agent_name,
                            "model": "unknown",
                            "tools": "unknown",
                            "debug_hooks": False
                        })
                except Exception as e:
                    agent_info.append({
                        "name": agent_name,
                        "model": "error",
                        "tools": "error",
                        "debug_hooks": False,
                        "error": str(e)
                    })

            if agent_info:
                st.dataframe(agent_info)
            else:
                st.info("No agent clients available")
        else:
            st.info("Agent client information not available")


def main():
    """Main function to run the Streamlit app."""
    ui = ResearchUI()
    ui.run()


if __name__ == "__main__":
    main()
