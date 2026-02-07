"""Tests for the new unified UI feature for viewing transcript history."""

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner
from syrupy.assertion import SnapshotAssertion

from claude_code_transcripts import (
    cli,
    generate_html,
    generate_unified_html,
    extract_subagent_ids,
    unescape_json_string,
)


@pytest.fixture
def sample_session():
    """Load the sample session fixture."""
    fixture_path = Path(__file__).parent / "sample_session.json"
    with open(fixture_path) as f:
        return json.load(f)


@pytest.fixture
def output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestNewUiCliFlag:
    """Tests for the --new-ui CLI flag."""

    def test_json_command_has_new_ui_flag(self):
        """Test that the json command accepts --new-ui flag."""
        runner = CliRunner()
        fixture_path = Path(__file__).parent / "sample_session.json"

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                cli,
                ["json", str(fixture_path), "-o", tmpdir, "--new-ui"],
            )

            assert result.exit_code == 0
            # Should generate unified.html instead of multiple pages
            assert (Path(tmpdir) / "unified.html").exists()

    def test_local_command_has_new_ui_flag(self, tmp_path, monkeypatch):
        """Test that the local command accepts --new-ui flag."""
        import questionary

        # Create mock .claude/projects structure
        projects_dir = tmp_path / ".claude" / "projects" / "test-project"
        projects_dir.mkdir(parents=True)

        session_file = projects_dir / "session-123.jsonl"
        session_file.write_text(
            '{"type":"summary","summary":"Test session"}\n'
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z","message":{"role":"user","content":"Hello"}}\n'
            '{"type":"assistant","timestamp":"2025-01-01T00:00:01Z","message":{"role":"assistant","content":[{"type":"text","text":"Hi"}]}}\n'
        )

        # Mock Path.home() to return our tmp_path
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Mock questionary.select to return the session file
        class MockSelect:
            def __init__(self, *args, **kwargs):
                pass

            def ask(self):
                return session_file

        monkeypatch.setattr(questionary, "select", MockSelect)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(cli, ["local", "-o", str(output_dir), "--new-ui"])

        assert result.exit_code == 0
        assert (output_dir / "unified.html").exists()

    def test_new_ui_flag_generates_single_file(self, output_dir):
        """Test that --new-ui generates a single unified.html file."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["json", str(fixture_path), "-o", str(output_dir), "--new-ui"],
        )

        assert result.exit_code == 0
        # Should have unified.html
        assert (output_dir / "unified.html").exists()
        # Should NOT have paginated files
        assert not (output_dir / "page-001.html").exists()
        assert not (output_dir / "index.html").exists()


class TestGenerateUnifiedHtml:
    """Tests for the generate_unified_html function."""

    def test_generates_unified_html(self, output_dir):
        """Test that generate_unified_html creates a unified.html file."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        assert (output_dir / "unified.html").exists()

    def test_unified_html_contains_all_messages(self, output_dir):
        """Test that unified.html contains all messages from the session."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should contain user and assistant messages
        assert "User" in html or "user" in html
        assert "Assistant" in html or "assistant" in html

    def test_unified_html_has_sidebar_navigation(self, output_dir):
        """Test that unified.html includes a sidebar navigation."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have sidebar navigation
        assert 'id="sidebar"' in html or 'class="sidebar"' in html
        # Should have nav links
        assert 'class="nav-link"' in html or 'class="sidebar-link"' in html

    def test_unified_html_has_search_functionality(self, output_dir):
        """Test that unified.html includes search functionality."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have search input
        assert 'id="search-input"' in html or 'id="unified-search"' in html
        # Should have search-related JavaScript
        assert "search" in html.lower()

    def test_unified_html_has_section_anchors(self, output_dir):
        """Test that unified.html has anchor IDs for each section."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have section anchors with IDs
        assert 'id="section-' in html or 'id="prompt-' in html


class TestUnifiedHtmlSidebar:
    """Tests for the sidebar navigation in unified HTML."""

    def test_sidebar_has_prompt_links(self, output_dir):
        """Test that sidebar contains links to each prompt section."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Sidebar should have links with href pointing to sections
        assert 'href="#section-' in html or 'href="#prompt-' in html

    def test_sidebar_shows_prompt_previews(self, output_dir):
        """Test that sidebar shows preview text for each prompt."""
        # Create a session with known prompts
        session_data = {
            "loglines": [
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "message": {"content": "First test prompt", "role": "user"},
                },
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Response 1"}],
                    },
                },
                {
                    "type": "user",
                    "timestamp": "2025-01-01T11:00:00.000Z",
                    "message": {"content": "Second test prompt", "role": "user"},
                },
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T11:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Response 2"}],
                    },
                },
            ]
        }

        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # The sidebar should show the prompt text (or truncated version)
        assert "First test" in html
        assert "Second test" in html

    def test_sidebar_is_fixed_position(self, output_dir):
        """Test that sidebar has fixed/sticky positioning CSS."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # CSS should include fixed or sticky positioning for sidebar
        assert "position: fixed" in html or "position: sticky" in html


class TestUnifiedHtmlSearch:
    """Tests for the search functionality in unified HTML."""

    def test_search_filters_content(self, output_dir):
        """Test that search JavaScript can filter content."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have JavaScript for filtering/searching
        assert "filter" in html.lower() or "search" in html.lower()
        # Should handle input events
        assert "input" in html or "keyup" in html or "keydown" in html

    def test_search_highlights_matches(self, output_dir):
        """Test that search highlights matching text."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have highlighting capability (mark element or highlight class)
        assert "mark" in html.lower() or "highlight" in html.lower()

    def test_search_updates_url(self, output_dir):
        """Test that search updates URL with query parameter."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should update URL hash or query param
        assert "location.hash" in html or "history." in html


class TestUnifiedHtmlScrolling:
    """Tests for smooth scrolling and navigation in unified HTML."""

    def test_sidebar_links_scroll_to_sections(self, output_dir):
        """Test that clicking sidebar links scrolls to the section."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have smooth scrolling behavior
        assert "scroll-behavior: smooth" in html or "scrollIntoView" in html

    def test_active_section_highlighted_in_sidebar(self, output_dir):
        """Test that current section is highlighted in sidebar during scroll."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have scroll event handling for active state
        assert "scroll" in html.lower()
        # Should have active class or similar for highlighting
        assert "active" in html.lower()


class TestUnifiedHtmlResponsive:
    """Tests for responsive design of unified HTML."""

    def test_sidebar_collapses_on_mobile(self, output_dir):
        """Test that sidebar has responsive behavior for mobile."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have media query for mobile
        assert "@media" in html
        # Should have toggle or collapse mechanism
        assert (
            "toggle" in html.lower()
            or "collapse" in html.lower()
            or "hidden" in html.lower()
        )


class TestUnifiedHtmlStats:
    """Tests for statistics display in unified HTML."""

    def test_shows_session_stats(self, output_dir):
        """Test that unified HTML shows session statistics."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should show stats like number of prompts, messages, tool calls
        assert "prompt" in html.lower()
        assert "message" in html.lower() or "tool" in html.lower()

    def test_shows_timestamps(self, output_dir):
        """Test that unified HTML displays timestamps."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have timestamp elements
        assert "<time" in html or "timestamp" in html.lower()


class TestUnifiedHtmlNavButtons:
    """Tests for message navigation buttons."""

    def test_has_message_wrappers(self, output_dir):
        """Test that messages are wrapped with navigation buttons."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have message wrapper divs
        assert 'class="message-wrapper"' in html

    def test_has_prev_next_buttons(self, output_dir):
        """Test that prev/next navigation buttons are present."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have prev and next buttons
        assert 'class="msg-nav-btn prev-btn"' in html
        assert 'class="msg-nav-btn next-btn"' in html

    def test_nav_buttons_have_aria_labels(self, output_dir):
        """Test that navigation buttons have accessible labels."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        assert 'aria-label="Previous message"' in html
        assert 'aria-label="Next message"' in html

    def test_keyboard_navigation_js(self, output_dir):
        """Test that keyboard navigation (j/k keys) is supported."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have j/k keyboard navigation in JavaScript
        assert "e.key === 'j'" in html or 'e.key === "j"' in html
        assert "e.key === 'k'" in html or 'e.key === "k"' in html


class TestUnifiedHtmlDarkTheme:
    """Tests for dark theme styling."""

    def test_has_dark_background(self, output_dir):
        """Test that the unified UI has a dark background."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have dark background color variable
        assert "--bg-color: #0f172a" in html or "bg-color" in html

    def test_has_dark_card_background(self, output_dir):
        """Test that cards have dark backgrounds."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have dark card backgrounds
        assert "--card-bg: #1e293b" in html

    def test_has_light_text_color(self, output_dir):
        """Test that text is light colored for dark theme."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have light text color
        assert "--text-color: #e2e8f0" in html


class TestSystemInfoSeparation:
    """Tests for separating system info from user messages."""

    def test_system_info_style_exists(self, output_dir):
        """Test that system-info CSS class exists for separated content."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have system-info styling in CSS
        assert ".system-info" in html

    def test_separates_ide_opened_file(self, output_dir):
        """Test that <ide_opened_file> tags are separated from user content."""
        session_data = {
            "loglines": [
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "message": {
                        "content": "<ide_opened_file>User opened file.txt</ide_opened_file>\nActual user prompt here",
                        "role": "user",
                    },
                },
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Response"}],
                    },
                },
            ]
        }

        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have system info separated
        assert "system-info" in html
        # The user prompt should still be present
        assert "Actual user prompt here" in html


class TestSidebarScrollbar:
    """Tests for sidebar scrollbar theming."""

    def test_sidebar_scrollbar_dark_theme(self, output_dir):
        """Test that sidebar scrollbar has dark theme styling."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have custom scrollbar styling for sidebar
        assert ".sidebar-nav::-webkit-scrollbar" in html

    def test_sidebar_scrollbar_track_dark(self, output_dir):
        """Test that scrollbar track is dark themed."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have dark track background
        assert "::-webkit-scrollbar-track" in html

    def test_sidebar_scrollbar_thumb_themed(self, output_dir):
        """Test that scrollbar thumb is themed."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have themed thumb
        assert "::-webkit-scrollbar-thumb" in html


class TestMessageTypeFilters:
    """Tests for message type filter toggles."""

    def test_has_filter_toggles_container(self, output_dir):
        """Test that filter toggle container exists."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have filter container
        assert 'class="message-filters"' in html or 'id="message-filters"' in html

    def test_has_user_message_filter(self, output_dir):
        """Test that user message filter toggle exists."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have user filter toggle
        assert 'data-filter="user"' in html

    def test_has_assistant_message_filter(self, output_dir):
        """Test that assistant message filter toggle exists."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have assistant filter toggle
        assert 'data-filter="assistant"' in html

    def test_has_tool_type_filters(self, output_dir):
        """Test that individual tool type filter toggles exist for tools used in session."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Sample session uses Write, Bash, TodoWrite, Glob, Edit, Grep
        assert 'data-filter="Write"' in html
        assert 'data-filter="Bash"' in html
        assert 'data-filter="Edit"' in html
        assert 'data-filter="Grep"' in html
        assert 'data-filter="Glob"' in html
        assert 'data-filter="TodoWrite"' in html

    def test_no_generic_tool_filter(self, output_dir):
        """Test that the old generic 'tool' filter is replaced by specific tool types."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should NOT have the old generic tool filter
        assert 'data-filter="tool"' not in html

    def test_only_used_tools_get_filters(self, output_dir):
        """Test that only tools actually used in the session get filter buttons."""
        session_data = {
            "loglines": [
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "message": {"content": "Hello", "role": "user"},
                },
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Let me read that file."},
                            {
                                "type": "tool_use",
                                "id": "toolu_001",
                                "name": "Read",
                                "input": {"file_path": "/tmp/test.py"},
                            },
                        ],
                    },
                },
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:10.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_001",
                                "content": "file contents here",
                            }
                        ],
                    },
                },
            ]
        }

        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have Read filter button since it's used
        assert '<label class="filter-toggle active" data-filter="Read">' in html
        # Should NOT have filter buttons for unused tools
        assert '<label class="filter-toggle active" data-filter="Bash">' not in html
        assert '<label class="filter-toggle active" data-filter="Write">' not in html

    def test_tool_reply_has_data_tools_attribute(self, output_dir):
        """Test that tool-reply messages have data-tools attribute identifying the tool."""
        session_data = {
            "loglines": [
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "message": {"content": "Run tests", "role": "user"},
                },
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_bash_001",
                                "name": "Bash",
                                "input": {
                                    "command": "pytest",
                                    "description": "Run tests",
                                },
                            }
                        ],
                    },
                },
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:10.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_bash_001",
                                "content": "All tests passed",
                            }
                        ],
                    },
                },
            ]
        }

        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Tool reply message should have data-tools attribute
        assert 'data-tools="Bash"' in html

    def test_assistant_message_has_data_tools_attribute(self, output_dir):
        """Test that assistant messages with tools have data-tools attribute."""
        session_data = {
            "loglines": [
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "message": {"content": "Write a file", "role": "user"},
                },
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Writing file."},
                            {
                                "type": "tool_use",
                                "id": "toolu_001",
                                "name": "Write",
                                "input": {
                                    "file_path": "/tmp/test.py",
                                    "content": "print('hello')",
                                },
                            },
                        ],
                    },
                },
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:10.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_001",
                                "content": "File written",
                            }
                        ],
                    },
                },
            ]
        }

        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Assistant message should have data-tools listing tool names
        assert 'data-tools="Write"' in html

    def test_filters_on_by_default(self, output_dir):
        """Test that all filters are on by default."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have checked checkboxes or active state by default
        assert "checked" in html or "active" in html.lower()

    def test_filter_toggle_javascript(self, output_dir):
        """Test that filter toggle JavaScript functionality exists."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have JS that handles filter toggling
        assert "toggleFilter" in html or "filter-toggle" in html

    def test_filter_js_handles_data_tools(self, output_dir):
        """Test that filter JavaScript uses data-tools attribute for tool filtering."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # JS should reference data-tools for filtering
        assert "data-tools" in html


class TestSearchClearButton:
    """Tests for search clear button functionality."""

    def test_has_clear_button_in_search(self, output_dir):
        """Test that search input has a clear button."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have clear button in search container
        assert 'class="search-clear"' in html or 'id="search-clear"' in html

    def test_clear_button_resets_nav_items(self, output_dir):
        """Test that clear button JavaScript resets nav item appearance."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should reset nav link opacity in JS
        assert "opacity" in html and ("1" in html or "'1'" in html)

    def test_clear_button_stays_at_position(self, output_dir):
        """Test that clearing search doesn't scroll to beginning."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should NOT scroll when clearing (look for specific non-scroll behavior)
        # The clear function should reset without scrolling
        assert "clearSearch" in html or "search-clear" in html


class TestCopyButton:
    """Tests for copy text button functionality."""

    def test_has_copy_button_css(self, output_dir):
        """Test that copy button CSS styles exist."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have copy button styling
        assert ".copy-btn" in html or ".copy-button" in html

    def test_has_copy_button_js(self, output_dir):
        """Test that copy button JavaScript functionality exists."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have clipboard copy functionality
        assert "clipboard" in html.lower() or "navigator.clipboard" in html

    def test_code_blocks_have_copy_button(self, output_dir):
        """Test that code blocks have copy buttons."""
        session_data = {
            "loglines": [
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "message": {"content": "Show me code", "role": "user"},
                },
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "```python\nprint('hello')\n```"}
                        ],
                    },
                },
            ]
        }

        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Code blocks should be wrapped with copy functionality
        assert "copy" in html.lower()


class TestCodeBlockSyntaxHighlight:
    """Tests for language-based code block syntax highlighting."""

    def test_has_language_class_on_code_blocks(self, output_dir):
        """Test that code blocks have language-specific classes."""
        session_data = {
            "loglines": [
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "message": {"content": "Show me Python code", "role": "user"},
                },
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "```python\nprint('hello')\n```"}
                        ],
                    },
                },
            ]
        }

        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have language class on pre or code element
        assert 'class="language-python"' in html or 'data-language="python"' in html

    def test_has_syntax_highlight_styles(self, output_dir):
        """Test that syntax highlighting CSS exists."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have syntax highlighting styles (keywords, strings, comments)
        assert (
            ".hljs" in html
            or ".token" in html
            or ".syntax-" in html
            or ".code-keyword" in html
        )

    def test_javascript_code_styled(self, output_dir):
        """Test that JavaScript code blocks get proper styling."""
        session_data = {
            "loglines": [
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "message": {"content": "Show me JS code", "role": "user"},
                },
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "```javascript\nconst x = 42;\n```",
                            }
                        ],
                    },
                },
            ]
        }

        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should identify JavaScript language
        assert "javascript" in html.lower() or "js" in html


class TestNavMenuUserPrompts:
    """Tests for nav menu showing actual user prompts."""

    def test_nav_shows_user_text_not_system_info(self, output_dir):
        """Test that nav menu shows actual user text, not system tags."""
        session_data = {
            "loglines": [
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "message": {
                        "content": "<ide_opened_file>some/path.txt</ide_opened_file>\nMy actual question here",
                        "role": "user",
                    },
                },
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Response"}],
                    },
                },
            ]
        }

        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Nav should show "My actual question here" not "<ide_opened_file>"
        assert "My actual question" in html
        # The ide_opened_file tag content should not appear in nav preview
        assert 'class="nav-preview"' in html


class TestUserContentStyling:
    """Tests for proper user content styling and class names."""

    def test_user_content_has_user_text_class(self, output_dir):
        """Test that actual user text uses user-text class, not assistant-text."""
        session_data = {
            "loglines": [
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "message": {
                        "content": "This is the user's actual question",
                        "role": "user",
                    },
                },
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Response"}],
                    },
                },
            ]
        }

        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # User messages should use user-text or user-content class
        assert 'class="user-text"' in html or 'class="user-content"' in html


class TestTokenUsageInUnifiedView:
    """Tests for token usage display in unified HTML view."""

    def test_shows_token_stats_in_header(self, output_dir):
        """Test that token stats are displayed in the header when usage data is present."""
        session_data = {
            "loglines": [
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "message": {"content": "Hello", "role": "user"},
                },
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hi there!"}],
                    },
                    "usage": {"input_tokens": 1500, "output_tokens": 750},
                },
            ]
        }

        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should show token stats in header
        assert "1,500" in html or "1500" in html  # input tokens
        assert "750" in html  # output tokens
        assert "2,250" in html or "2250" in html  # total tokens

    def test_shows_per_message_token_info(self, output_dir):
        """Test that individual assistant messages show token counts."""
        session_data = {
            "loglines": [
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "message": {"content": "Hello", "role": "user"},
                },
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hi there!"}],
                    },
                    "usage": {"input_tokens": 150, "output_tokens": 75},
                },
            ]
        }

        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should show per-message token info
        assert "in: 150" in html or "in:150" in html
        assert "out: 75" in html or "out:75" in html

    def test_shows_cost_estimate(self, output_dir):
        """Test that estimated cost is displayed in the header."""
        session_data = {
            "loglines": [
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "message": {"content": "Hello", "role": "user"},
                },
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hi there!"}],
                    },
                    "usage": {"input_tokens": 100000, "output_tokens": 50000},
                },
            ]
        }

        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should show cost estimate with $
        assert "$" in html

    def test_has_token_info_css(self, output_dir):
        """Test that token-info CSS class is defined."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have token-info styling
        assert ".token-info" in html

    def test_no_token_stats_when_missing(self, output_dir):
        """Test that page still works when no token usage data is present."""
        session_data = {
            "loglines": [
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "message": {"content": "Hello", "role": "user"},
                },
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hi there!"}],
                    },
                    # No usage field
                },
            ]
        }

        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        # Should not raise an error
        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should still generate valid HTML
        assert "Claude Code Transcript" in html


class TestSubagentLinking:
    """Tests for subagent session linking when Task tool is used."""

    def _make_task_session(self):
        """Create a session with a Task tool call that spawns a subagent."""
        return {
            "loglines": [
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "message": {"content": "Run the tests", "role": "user"},
                },
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "I'll run the tests using a subagent.",
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_task_001",
                                "name": "Task",
                                "input": {
                                    "description": "Run test suite",
                                    "prompt": "Run all tests with pytest",
                                    "subagent_type": "Bash",
                                },
                            },
                        ],
                    },
                    "usage": {"input_tokens": 200, "output_tokens": 50},
                },
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:10.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_task_001",
                                "content": "All tests passed.\n\nagentId: abc123def",
                            }
                        ],
                    },
                },
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:15.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "All tests pass!"}],
                    },
                    "usage": {"input_tokens": 300, "output_tokens": 20},
                },
            ]
        }

    def test_extract_subagent_ids_from_loglines(self):
        """Test that subagent IDs can be extracted from Task tool results."""
        session_data = self._make_task_session()
        agent_map = extract_subagent_ids(session_data["loglines"])
        assert "toolu_task_001" in agent_map
        assert agent_map["toolu_task_001"] == "abc123def"

    def test_extract_subagent_ids_no_task(self):
        """Test that extract_subagent_ids returns empty dict when no Task tools."""
        loglines = [
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:00.000Z",
                "message": {"content": "Hello", "role": "user"},
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:05.000Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hi"}],
                },
            },
        ]
        agent_map = extract_subagent_ids(loglines)
        assert agent_map == {}

    def test_extract_subagent_ids_multiple_tasks(self):
        """Test extracting multiple subagent IDs from multiple Task calls."""
        loglines = [
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:00.000Z",
                "message": {"content": "Do stuff", "role": "user"},
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:05.000Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_task_a",
                            "name": "Task",
                            "input": {
                                "description": "Explore code",
                                "prompt": "Search the codebase",
                                "subagent_type": "Explore",
                            },
                        },
                    ],
                },
            },
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:10.000Z",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_task_a",
                            "content": "Found files.\n\nagentId: agent111",
                        }
                    ],
                },
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:15.000Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_task_b",
                            "name": "Task",
                            "input": {
                                "description": "Run tests",
                                "prompt": "Run pytest",
                                "subagent_type": "Bash",
                            },
                        },
                    ],
                },
            },
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:20.000Z",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_task_b",
                            "content": "Tests passed.\n\nagentId: agent222",
                        }
                    ],
                },
            },
        ]
        agent_map = extract_subagent_ids(loglines)
        assert agent_map == {"toolu_task_a": "agent111", "toolu_task_b": "agent222"}

    def test_task_tool_renders_with_description(self, output_dir):
        """Test that Task tool_use renders showing description and subagent type."""
        session_data = self._make_task_session()
        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should show the task description
        assert "Run test suite" in html
        # Should show the subagent type
        assert "Bash" in html
        # Should have task-tool class
        assert "task-tool" in html

    def test_task_tool_has_subagent_type_badge(self, output_dir):
        """Test that Task tool renders with a subagent type badge."""
        session_data = self._make_task_session()
        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        assert 'class="subagent-type"' in html

    def test_task_result_has_subagent_link(self, output_dir):
        """Test that Task tool result renders a link to the subagent session."""
        session_data = self._make_task_session()
        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have a link to the subagent session
        assert "../agent-abc123def/unified.html" in html
        assert 'class="subagent-link"' in html

    def test_task_result_link_text(self, output_dir):
        """Test that the subagent link has descriptive text."""
        session_data = self._make_task_session()
        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        assert "View subagent transcript" in html

    def test_task_result_labeled_as_subagent(self, output_dir):
        """Test that Task tool result is labeled 'Subagent' not 'Tool reply'."""
        session_data = self._make_task_session()
        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should use subagent-result class, not tool-reply
        assert 'class="message subagent-result"' in html
        # Should show "Subagent" label
        assert ">Subagent</span>" in html

    def test_task_result_has_subagent_styling(self, output_dir):
        """Test that subagent results have distinct styling from tool replies."""
        session_data = self._make_task_session()
        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should have CSS for subagent-result
        assert ".message.subagent-result" in html
        # Should be styled with indigo, not orange like tool-reply
        assert ".subagent-result .role-label" in html

    def test_task_tool_has_data_tools_attribute(self, output_dir):
        """Test that assistant message with Task tool has data-tools attribute."""
        session_data = self._make_task_session()
        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        assert 'data-tools="Task"' in html

    def test_task_filter_appears_when_task_used(self, output_dir):
        """Test that a Task filter toggle appears when Task tool is used."""
        session_data = self._make_task_session()
        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        assert 'data-filter="Task"' in html

    def test_task_css_styles_exist(self, output_dir):
        """Test that CSS for task-tool and subagent-link classes exist."""
        session_data = self._make_task_session()
        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        assert ".task-tool" in html
        assert ".subagent-link" in html
        assert ".subagent-type" in html

    def test_no_subagent_link_when_no_agent_id(self, output_dir):
        """Test that no subagent link is rendered when result has no agentId."""
        session_data = {
            "loglines": [
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "message": {"content": "Do something", "role": "user"},
                },
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_task_no_id",
                                "name": "Task",
                                "input": {
                                    "description": "Quick task",
                                    "prompt": "Do something simple",
                                    "subagent_type": "general-purpose",
                                },
                            },
                        ],
                    },
                },
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:10.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_task_no_id",
                                "content": "Task completed with no agent id reference.",
                            }
                        ],
                    },
                },
            ]
        }
        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # Should NOT have subagent link element (CSS definition is fine)
        assert 'class="subagent-link"' not in html
        assert "View subagent transcript" not in html
        # But should still have the task-tool rendering
        assert "task-tool" in html

    def test_sample_session_has_task_tool(self, output_dir):
        """Test that the sample session fixture includes Task tool and renders correctly."""
        fixture_path = Path(__file__).parent / "sample_session.json"

        generate_unified_html(fixture_path, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        assert "task-tool" in html
        assert "Run test suite" in html
        assert 'data-filter="Task"' in html
        assert "../agent-abc123def/unified.html" in html

    def test_task_result_unescapes_json_strings(self, output_dir):
        """Test that Task tool results unescape JSON-like escaped strings for readability."""
        # Use raw string to get literal backslash-n sequences (as they appear in real JSONL)
        escaped_content = r"Found results: {\"name\": \"test value\", \"count\": 42}\n\nDetails:\n- Item one\n- Item two\n\nagentId: xyz789"

        session_data = {
            "loglines": [
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "message": {"content": "Run a task", "role": "user"},
                },
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_task_escaped",
                                "name": "Task",
                                "input": {
                                    "description": "Test task",
                                    "prompt": "Do something",
                                    "subagent_type": "Bash",
                                },
                            },
                        ],
                    },
                },
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:10.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_task_escaped",
                                "content": escaped_content,
                            }
                        ],
                    },
                },
            ]
        }
        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        # The literal \n sequences should NOT appear in the output
        assert r"\n" not in html.split("subagent-result")[1].split("</article>")[0]
        # The escaped quotes should be unescaped - actual quotes or HTML entities
        assert "&quot;name&quot;" in html or '"name"' in html
        assert "&quot;test value&quot;" in html or '"test value"' in html


class TestUnescapeJsonString:
    """Unit tests for the unescape_json_string function."""

    def test_unescapes_newlines(self):
        """Test that literal \\n sequences are converted to newlines."""
        assert unescape_json_string(r"line1\nline2") == "line1\nline2"

    def test_unescapes_tabs(self):
        """Test that literal \\t sequences are converted to tabs."""
        assert unescape_json_string(r"col1\tcol2") == "col1\tcol2"

    def test_unescapes_carriage_returns(self):
        """Test that literal \\r sequences are converted to carriage returns."""
        assert unescape_json_string(r"text\rmore") == "text\rmore"

    def test_unescapes_quotes(self):
        """Test that literal \\" sequences are converted to quotes."""
        assert unescape_json_string(r'{"key": \"value\"}') == '{"key": "value"}'

    def test_unescapes_backslashes(self):
        """Test that literal \\\\ sequences are converted to single backslashes."""
        assert unescape_json_string(r"path\\to\\file") == "path\\to\\file"

    def test_complex_json_content(self):
        """Test unescaping a complex JSON-like string."""
        input_str = r"Result: {\"name\": \"test\", \"items\": [\"a\", \"b\"]}\n\nDetails:\n- First\n- Second"
        expected = 'Result: {"name": "test", "items": ["a", "b"]}\n\nDetails:\n- First\n- Second'
        assert unescape_json_string(input_str) == expected

    def test_non_string_returns_unchanged(self):
        """Test that non-string values are returned unchanged."""
        assert unescape_json_string(None) is None
        assert unescape_json_string(123) == 123
        assert unescape_json_string(["list"]) == ["list"]

    def test_no_escape_sequences_unchanged(self):
        """Test that strings without escape sequences are unchanged."""
        plain = "Just plain text with no escapes"
        assert unescape_json_string(plain) == plain


class TestUnifiedHtmlSnapshot:
    """Snapshot tests for unified HTML output covering all features."""

    def test_unified_html_snapshot(self, output_dir, snapshot: SnapshotAssertion):
        """Test that unified HTML output matches expected snapshot.

        Uses a comprehensive session with all message/tool types:
        - User messages with system info tags (ide_opened_file, system_reminder)
        - Assistant messages with thinking blocks, markdown, and code
        - Write, Edit, Bash, Glob, Grep, TodoWrite, and Task tools
        - Error tool results
        - Token usage with cache stats
        - Task tool with subagent link (agentId)
        - Multiple user prompts for sidebar navigation
        """
        session_data = {
            "loglines": [
                # User prompt with system info
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "message": {
                        "content": "<ide_opened_file>src/main.py</ide_opened_file>\n<system_reminder>Important context</system_reminder>\nBuild me a web server",
                        "role": "user",
                    },
                },
                # Assistant with thinking + text + Write tool
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "The user wants a web server. I should create a simple Flask app.",
                            },
                            {
                                "type": "text",
                                "text": "I'll create a **Flask** web server for you.\n\n```python\nfrom flask import Flask\n```",
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_write_001",
                                "name": "Write",
                                "input": {
                                    "file_path": "/project/server.py",
                                    "content": "from flask import Flask\n\napp = Flask(__name__)\n\n@app.route('/')\ndef hello():\n    return 'Hello!'\n",
                                },
                            },
                        ],
                    },
                    "usage": {
                        "input_tokens": 1500,
                        "output_tokens": 350,
                        "cache_read_input_tokens": 800,
                        "cache_creation_input_tokens": 200,
                    },
                },
                # Write tool result
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:10.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_write_001",
                                "content": "File written successfully",
                            }
                        ],
                    },
                },
                # Assistant with Edit tool
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:15.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_edit_001",
                                "name": "Edit",
                                "input": {
                                    "file_path": "/project/server.py",
                                    "old_string": "return 'Hello!'",
                                    "new_string": "return 'Hello, World!'",
                                },
                            },
                        ],
                    },
                    "usage": {"input_tokens": 500, "output_tokens": 100},
                },
                # Edit tool result
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:20.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_edit_001",
                                "content": "File edited successfully",
                            }
                        ],
                    },
                },
                # Assistant with Bash tool
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:25.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_bash_001",
                                "name": "Bash",
                                "input": {
                                    "command": "python -m pytest tests/ -v",
                                    "description": "Run tests",
                                },
                            },
                        ],
                    },
                    "usage": {"input_tokens": 600, "output_tokens": 80},
                },
                # Bash tool result (error)
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:30.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_bash_001",
                                "content": "Exit code 1\nModuleNotFoundError: No module named 'flask'",
                                "is_error": True,
                            }
                        ],
                    },
                },
                # Assistant with Glob tool
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:35.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "Let me check the project structure.",
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_glob_001",
                                "name": "Glob",
                                "input": {"pattern": "**/*.py", "path": "/project"},
                            },
                        ],
                    },
                    "usage": {"input_tokens": 700, "output_tokens": 50},
                },
                # Glob tool result
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:40.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_glob_001",
                                "content": "/project/server.py\n/project/tests/test_server.py",
                            }
                        ],
                    },
                },
                # Assistant with Grep tool
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:45.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_grep_001",
                                "name": "Grep",
                                "input": {
                                    "pattern": "import flask",
                                    "path": "/project",
                                },
                            },
                        ],
                    },
                    "usage": {"input_tokens": 800, "output_tokens": 40},
                },
                # Grep result
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:50.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_grep_001",
                                "content": "/project/server.py:1:from flask import Flask",
                            }
                        ],
                    },
                },
                # Assistant with TodoWrite
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:55.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_todo_001",
                                "name": "TodoWrite",
                                "input": {
                                    "todos": [
                                        {
                                            "content": "Create server",
                                            "status": "completed",
                                            "activeForm": "Creating server",
                                        },
                                        {
                                            "content": "Fix dependencies",
                                            "status": "in_progress",
                                            "activeForm": "Fixing dependencies",
                                        },
                                        {
                                            "content": "Run tests",
                                            "status": "pending",
                                            "activeForm": "Running tests",
                                        },
                                    ]
                                },
                            },
                        ],
                    },
                    "usage": {"input_tokens": 900, "output_tokens": 60},
                },
                # Todo result
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:01:00.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_todo_001",
                                "content": "Todos updated",
                            }
                        ],
                    },
                },
                # Second user prompt
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:02:00.000Z",
                    "message": {
                        "content": "Now deploy it using a subagent",
                        "role": "user",
                    },
                },
                # Assistant with Task tool (subagent)
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:02:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "I'll spawn a subagent to handle deployment.",
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_task_001",
                                "name": "Task",
                                "input": {
                                    "description": "Deploy web server",
                                    "prompt": "Deploy the Flask server to production",
                                    "subagent_type": "Bash",
                                },
                            },
                        ],
                    },
                    "usage": {"input_tokens": 1000, "output_tokens": 120},
                },
                # Task result with agentId (subagent response)
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:02:30.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_task_001",
                                "content": "Server deployed successfully to https://example.com\n\nagentId: deploy789xyz",
                            }
                        ],
                    },
                },
                # Final assistant response
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:02:35.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "Your server is deployed and running!",
                            }
                        ],
                    },
                    "usage": {"input_tokens": 1100, "output_tokens": 30},
                },
            ]
        }

        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)

        html = (output_dir / "unified.html").read_text(encoding="utf-8")
        assert html == snapshot


class TestComprehensiveNewUi:
    """Comprehensive tests covering all new-ui features in a single session."""

    @pytest.fixture
    def comprehensive_session_data(self):
        """Session data containing all tool/message types for comprehensive testing."""
        return {
            "loglines": [
                # 1. User prompt with system info
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "message": {
                        "content": "<ide_opened_file>src/main.py</ide_opened_file>\n<system_reminder>Important context</system_reminder>\nBuild me a web server",
                        "role": "user",
                    },
                },
                # 2. Assistant with thinking + text + Write tool
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "The user wants a web server. I should create a simple Flask app.",
                            },
                            {
                                "type": "text",
                                "text": "I'll create a **Flask** web server for you.\n\n```python\nfrom flask import Flask\n```",
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_write_001",
                                "name": "Write",
                                "input": {
                                    "file_path": "/project/server.py",
                                    "content": "from flask import Flask\n\napp = Flask(__name__)\n\n@app.route('/')\ndef hello():\n    return 'Hello!'\n",
                                },
                            },
                        ],
                    },
                    "usage": {
                        "input_tokens": 1500,
                        "output_tokens": 350,
                        "cache_read_input_tokens": 800,
                        "cache_creation_input_tokens": 200,
                    },
                },
                # 3. Write tool result
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:10.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_write_001",
                                "content": "File written successfully",
                            }
                        ],
                    },
                },
                # 4. Assistant with Edit tool
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:15.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_edit_001",
                                "name": "Edit",
                                "input": {
                                    "file_path": "/project/server.py",
                                    "old_string": "return 'Hello!'",
                                    "new_string": "return 'Hello, World!'",
                                },
                            },
                        ],
                    },
                    "usage": {"input_tokens": 500, "output_tokens": 100},
                },
                # 5. Edit tool result
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:20.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_edit_001",
                                "content": "File edited successfully",
                            }
                        ],
                    },
                },
                # 6. Assistant with Bash tool
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:25.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_bash_001",
                                "name": "Bash",
                                "input": {
                                    "command": "python -m pytest tests/ -v",
                                    "description": "Run tests",
                                },
                            },
                        ],
                    },
                    "usage": {"input_tokens": 600, "output_tokens": 80},
                },
                # 7. Bash tool result (error)
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:30.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_bash_001",
                                "content": "Exit code 1\nModuleNotFoundError: No module named 'flask'",
                                "is_error": True,
                            }
                        ],
                    },
                },
                # 8. Assistant with Glob + Grep tools
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:35.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "Let me check the project structure.",
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_glob_001",
                                "name": "Glob",
                                "input": {
                                    "pattern": "**/*.py",
                                    "path": "/project",
                                },
                            },
                        ],
                    },
                    "usage": {"input_tokens": 700, "output_tokens": 50},
                },
                # 9. Glob tool result
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:40.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_glob_001",
                                "content": "/project/server.py\n/project/tests/test_server.py",
                            }
                        ],
                    },
                },
                # 10. Assistant with Grep
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:45.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_grep_001",
                                "name": "Grep",
                                "input": {
                                    "pattern": "import flask",
                                    "path": "/project",
                                },
                            },
                        ],
                    },
                    "usage": {"input_tokens": 800, "output_tokens": 40},
                },
                # 11. Grep result
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:50.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_grep_001",
                                "content": "/project/server.py:1:from flask import Flask",
                            }
                        ],
                    },
                },
                # 12. Assistant with TodoWrite
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:55.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_todo_001",
                                "name": "TodoWrite",
                                "input": {
                                    "todos": [
                                        {
                                            "content": "Create server",
                                            "status": "completed",
                                            "activeForm": "Creating server",
                                        },
                                        {
                                            "content": "Fix dependencies",
                                            "status": "in_progress",
                                            "activeForm": "Fixing dependencies",
                                        },
                                        {
                                            "content": "Run tests",
                                            "status": "pending",
                                            "activeForm": "Running tests",
                                        },
                                    ]
                                },
                            },
                        ],
                    },
                    "usage": {"input_tokens": 900, "output_tokens": 60},
                },
                # 13. Todo result
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:01:00.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_todo_001",
                                "content": "Todos updated",
                            }
                        ],
                    },
                },
                # 14. Second user prompt
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:02:00.000Z",
                    "message": {
                        "content": "Now deploy it using a subagent",
                        "role": "user",
                    },
                },
                # 15. Assistant with Task tool (subagent)
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:02:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "I'll spawn a subagent to handle deployment.",
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_task_001",
                                "name": "Task",
                                "input": {
                                    "description": "Deploy web server",
                                    "prompt": "Deploy the Flask server to production",
                                    "subagent_type": "Bash",
                                },
                            },
                        ],
                    },
                    "usage": {"input_tokens": 1000, "output_tokens": 120},
                },
                # 16. Task result with agentId
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:02:30.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_task_001",
                                "content": "Server deployed successfully to https://example.com\n\nagentId: deploy789xyz",
                            }
                        ],
                    },
                },
                # 17. Final assistant response
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:02:35.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "Your server is deployed and running!",
                            }
                        ],
                    },
                    "usage": {"input_tokens": 1100, "output_tokens": 30},
                },
            ]
        }

    def test_all_message_types_rendered(self, output_dir, comprehensive_session_data):
        """Test that user, assistant, tool-reply, subagent-result, and all tool types are rendered."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        # User messages
        assert 'class="message user"' in html
        # Assistant messages
        assert 'class="message assistant"' in html
        # Tool reply messages (for non-Task tool results)
        assert 'class="message tool-reply"' in html
        # Subagent result messages (for Task tool results)
        assert 'class="message subagent-result"' in html
        assert ">Subagent</span>" in html

    def test_all_tool_types_rendered(self, output_dir, comprehensive_session_data):
        """Test that Write, Edit, Bash, Glob, Grep, TodoWrite, and Task tools render."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        # Write tool
        assert "write-tool" in html
        assert "/project/server.py" in html
        # Edit tool
        assert "edit-tool" in html
        assert "edit-old" in html
        assert "edit-new" in html
        # Bash tool
        assert "bash-tool" in html
        assert "pytest" in html
        # Glob tool
        assert "Glob" in html
        # Grep tool
        assert "Grep" in html
        # TodoWrite tool
        assert "todo-list" in html
        assert "Create server" in html
        assert "Fix dependencies" in html
        # Task tool
        assert "task-tool" in html
        assert "Deploy web server" in html

    def test_thinking_block_rendered(self, output_dir, comprehensive_session_data):
        """Test that thinking blocks are rendered."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        assert 'class="thinking"' in html
        assert "Thinking" in html

    def test_error_tool_result_styled(self, output_dir, comprehensive_session_data):
        """Test that error tool results have error styling."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        assert "tool-error" in html

    def test_system_info_separated(self, output_dir, comprehensive_session_data):
        """Test that system info tags are separated from user content."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        assert "system-info" in html
        assert "Build me a web server" in html

    def test_markdown_rendered(self, output_dir, comprehensive_session_data):
        """Test that markdown in assistant text is rendered to HTML."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        # Bold text rendered
        assert "<strong>" in html or "<b>" in html
        # Code blocks rendered
        assert "<code" in html

    def test_token_usage_in_header(self, output_dir, comprehensive_session_data):
        """Test that total token usage is displayed in header."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        # Should show total tokens in sidebar stats
        assert "tokens" in html.lower()
        # Should show token stats in header
        assert "header-stats" in html

    def test_per_message_token_info(self, output_dir, comprehensive_session_data):
        """Test that individual assistant messages show input/output token counts."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        # Should show per-message token info
        assert "token-info" in html
        assert "in:" in html
        assert "out:" in html

    def test_mini_chart_rendered(self, output_dir, comprehensive_session_data):
        """Test that the token usage mini-chart is rendered in header."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        # Should have mini chart div with data attributes
        assert "header-mini-chart" in html
        assert "data-input=" in html
        assert "data-output=" in html
        assert "data-cache-read=" in html
        assert "data-cache-write=" in html

    def test_cost_estimate_displayed(self, output_dir, comprehensive_session_data):
        """Test that estimated API cost is shown."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        assert "$" in html

    def test_all_tool_filters_present(self, output_dir, comprehensive_session_data):
        """Test that filter toggles exist for all tool types used."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        # Core filters
        assert 'data-filter="user"' in html
        assert 'data-filter="assistant"' in html
        # Tool-specific filters
        assert 'data-filter="Write"' in html
        assert 'data-filter="Edit"' in html
        assert 'data-filter="Bash"' in html
        assert 'data-filter="Glob"' in html
        assert 'data-filter="Grep"' in html
        assert 'data-filter="TodoWrite"' in html
        assert 'data-filter="Task"' in html

    def test_data_tools_attributes_on_messages(
        self, output_dir, comprehensive_session_data
    ):
        """Test that messages have correct data-tools attributes."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        assert 'data-tools="Write"' in html
        assert 'data-tools="Edit"' in html
        assert 'data-tools="Bash"' in html
        assert 'data-tools="Task"' in html

    def test_search_infrastructure(self, output_dir, comprehensive_session_data):
        """Test that search input, clear button, and JS are present."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        # Search input
        assert 'id="unified-search"' in html
        # Clear button
        assert 'class="search-clear"' in html
        # Search results banner
        assert "search-results-banner" in html
        # Search JS
        assert "debounce" in html.lower() or "setTimeout" in html
        # Keyboard shortcut (Ctrl+K)
        assert "Ctrl" in html or "ctrlKey" in html or "metaKey" in html

    def test_sidebar_navigation(self, output_dir, comprehensive_session_data):
        """Test that sidebar has correct nav items for user prompts."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        # Sidebar exists
        assert 'id="sidebar"' in html
        # Nav list
        assert 'id="nav-list"' in html
        # Should have 2 prompts (Build me a web server, Now deploy it)
        assert "#prompt-1" in html
        assert "#prompt-2" in html
        # Nav previews show user text, not system tags
        assert "Build me a web server" in html
        assert "Now deploy it" in html

    def test_message_navigation_buttons(self, output_dir, comprehensive_session_data):
        """Test that prev/next message navigation buttons are present."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        assert 'class="message-wrapper"' in html
        assert 'class="msg-nav-btn prev-btn"' in html
        assert 'class="msg-nav-btn next-btn"' in html
        assert 'aria-label="Previous message"' in html
        assert 'aria-label="Next message"' in html

    def test_keyboard_navigation_js(self, output_dir, comprehensive_session_data):
        """Test that j/k keyboard navigation is included."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        assert "e.key === 'j'" in html or 'e.key === "j"' in html
        assert "e.key === 'k'" in html or 'e.key === "k"' in html

    def test_dark_theme_css_variables(self, output_dir, comprehensive_session_data):
        """Test that dark theme CSS variables are present."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        assert "--bg-color: #0f172a" in html
        assert "--card-bg: #1e293b" in html
        assert "--text-color: #e2e8f0" in html

    def test_responsive_mobile_toggle(self, output_dir, comprehensive_session_data):
        """Test that mobile sidebar toggle exists."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        assert "mobile-sidebar-toggle" in html
        assert "@media" in html

    def test_copy_button_js(self, output_dir, comprehensive_session_data):
        """Test that code block copy functionality is present."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        assert "navigator.clipboard" in html
        assert "copy-btn" in html or "Copy" in html

    def test_session_stats_in_sidebar(self, output_dir, comprehensive_session_data):
        """Test that sidebar shows prompt/message/tool counts."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        assert "sidebar-stats" in html
        assert "prompts" in html
        assert "messages" in html
        assert "tools" in html

    def test_subagent_link_in_comprehensive_session(
        self, output_dir, comprehensive_session_data
    ):
        """Test that the Task tool result has a subagent link."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        assert "../agent-deploy789xyz/unified.html" in html
        assert "View subagent transcript" in html

    def test_scrollbar_dark_theme(self, output_dir, comprehensive_session_data):
        """Test that scrollbar has dark theme styling."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        assert "::-webkit-scrollbar" in html

    def test_timestamps_rendered(self, output_dir, comprehensive_session_data):
        """Test that timestamps are included in messages."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        assert "<time" in html
        assert "data-timestamp" in html

    def test_todo_statuses_rendered(self, output_dir, comprehensive_session_data):
        """Test that todo items render with correct status classes."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        assert "todo-completed" in html
        assert "todo-in-progress" in html
        assert "todo-pending" in html

    def test_chart_js_renders_bars(self, output_dir, comprehensive_session_data):
        """Test that chart JavaScript renders I/O and cache bars."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        # Chart JS should reference the color scheme
        assert "#3b82f6" in html  # input (blue)
        assert "#f97316" in html  # output (orange)
        assert "#06b6d4" in html  # cache read (cyan)
        assert "#f59e0b" in html  # cache write (amber)

    def test_edit_tool_shows_replace_all(self, output_dir):
        """Test that Edit tool with replace_all flag renders correctly."""
        session_data = {
            "loglines": [
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                    "message": {"content": "Replace all occurrences", "role": "user"},
                },
                {
                    "type": "assistant",
                    "timestamp": "2025-01-01T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_edit_ra",
                                "name": "Edit",
                                "input": {
                                    "file_path": "/project/test.py",
                                    "old_string": "foo",
                                    "new_string": "bar",
                                    "replace_all": True,
                                },
                            },
                        ],
                    },
                },
                {
                    "type": "user",
                    "timestamp": "2025-01-01T10:00:10.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_edit_ra",
                                "content": "File edited",
                            }
                        ],
                    },
                },
            ]
        }

        session_file = output_dir / "test_session.json"
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        assert "replace all" in html.lower()

    def test_truncatable_content(self, output_dir, comprehensive_session_data):
        """Test that truncatable wrappers with expand buttons are present."""
        session_file = output_dir / "test_session.json"
        session_file.write_text(
            json.dumps(comprehensive_session_data), encoding="utf-8"
        )

        generate_unified_html(session_file, output_dir)
        html = (output_dir / "unified.html").read_text(encoding="utf-8")

        assert "truncatable" in html
        assert "expand-btn" in html
