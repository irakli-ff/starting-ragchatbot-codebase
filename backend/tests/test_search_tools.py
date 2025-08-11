import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import SearchResults, VectorStore


class TestCourseSearchTool:
    """Test the CourseSearchTool functionality"""

    def test_tool_definition(self):
        """Test that tool definition is properly formatted"""
        mock_store = Mock(spec=VectorStore)
        tool = CourseSearchTool(mock_store)

        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]

    def test_execute_with_results(self, sample_search_results):
        """Test execute method with successful search results"""
        mock_store = Mock(spec=VectorStore)
        mock_store.search.return_value = sample_search_results
        mock_store.get_lesson_link.return_value = "https://example.com/lesson1"
        mock_store.get_course_link.return_value = "https://example.com/course"

        tool = CourseSearchTool(mock_store)

        result = tool.execute(query="Python programming")

        # Verify search was called
        mock_store.search.assert_called_once_with(
            query="Python programming", course_name=None, lesson_number=None
        )

        # Check result formatting
        assert result is not None
        assert "Python" in result
        assert len(tool.last_sources) > 0

    def test_execute_with_course_filter(self):
        """Test execute with course name filter"""
        mock_store = Mock(spec=VectorStore)
        mock_results = SearchResults(
            documents=["Content about Python"],
            metadata=[{"course_title": "Python Course", "lesson_number": 1}],
            distances=[0.1],
        )
        mock_store.search.return_value = mock_results
        mock_store.get_lesson_link.return_value = None
        mock_store.get_course_link.return_value = None

        tool = CourseSearchTool(mock_store)

        result = tool.execute(query="variables", course_name="Python")

        mock_store.search.assert_called_once_with(
            query="variables", course_name="Python", lesson_number=None
        )

        assert "Python Course" in result

    def test_execute_with_lesson_filter(self):
        """Test execute with lesson number filter"""
        mock_store = Mock(spec=VectorStore)
        mock_results = SearchResults(
            documents=["Lesson 2 content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 2}],
            distances=[0.1],
        )
        mock_store.search.return_value = mock_results
        mock_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_store)

        result = tool.execute(query="control flow", lesson_number=2)

        mock_store.search.assert_called_once_with(
            query="control flow", course_name=None, lesson_number=2
        )

        assert "Lesson 2" in result

    def test_execute_with_error(self):
        """Test execute handles search errors"""
        mock_store = Mock(spec=VectorStore)
        error_result = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="No course found matching 'NonExistent'",
        )
        mock_store.search.return_value = error_result

        tool = CourseSearchTool(mock_store)

        result = tool.execute(query="test", course_name="NonExistent")

        assert "No course found" in result
        assert len(tool.last_sources) == 0

    def test_execute_empty_results(self):
        """Test execute with empty search results"""
        mock_store = Mock(spec=VectorStore)
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_store.search.return_value = empty_results

        tool = CourseSearchTool(mock_store)

        result = tool.execute(query="xyz123")

        assert "No relevant content found" in result
        assert len(tool.last_sources) == 0

    def test_source_tracking_with_links(self):
        """Test that sources are properly tracked with links"""
        mock_store = Mock(spec=VectorStore)
        mock_results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": None},
            ],
            distances=[0.1, 0.2],
        )
        mock_store.search.return_value = mock_results
        mock_store.get_lesson_link.side_effect = ["https://lesson1.com", None]
        mock_store.get_course_link.return_value = "https://courseb.com"

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="test")

        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "Course A - Lesson 1"
        assert tool.last_sources[0]["link"] == "https://lesson1.com"
        assert tool.last_sources[1]["text"] == "Course B"
        assert tool.last_sources[1]["link"] == "https://courseb.com"


class TestCourseOutlineTool:
    """Test the CourseOutlineTool functionality"""

    def test_tool_definition(self):
        """Test that tool definition is properly formatted"""
        mock_store = Mock(spec=VectorStore)
        tool = CourseOutlineTool(mock_store)

        definition = tool.get_tool_definition()

        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["course_title"]

    def test_execute_with_course(self):
        """Test execute method returns course outline"""
        mock_store = Mock(spec=VectorStore)
        mock_store._resolve_course_name.return_value = "Test Course"

        # Mock course catalog response
        mock_catalog = Mock()
        mock_store.course_catalog = mock_catalog

        lessons_data = [
            {
                "lesson_number": 0,
                "lesson_title": "Intro",
                "lesson_link": "https://intro.com",
            },
            {"lesson_number": 1, "lesson_title": "Basics", "lesson_link": None},
        ]
        mock_catalog.get.return_value = {
            "metadatas": [
                {
                    "title": "Test Course",
                    "instructor": "John Doe",
                    "course_link": "https://course.com",
                    "lessons_json": json.dumps(lessons_data),
                }
            ]
        }

        tool = CourseOutlineTool(mock_store)
        result = tool.execute(course_title="Test")

        assert "Test Course" in result
        assert "John Doe" in result
        assert "https://course.com" in result
        assert "Lesson 0: Intro" in result
        assert "Lesson 1: Basics" in result
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["link"] == "https://course.com"

    def test_execute_course_not_found(self):
        """Test execute when course is not found"""
        mock_store = Mock(spec=VectorStore)
        mock_store._resolve_course_name.return_value = None

        tool = CourseOutlineTool(mock_store)
        result = tool.execute(course_title="NonExistent")

        assert "No course found" in result
        assert len(tool.last_sources) == 0

    def test_execute_with_exception(self):
        """Test execute handles exceptions gracefully"""
        mock_store = Mock(spec=VectorStore)
        mock_store._resolve_course_name.return_value = "Test Course"
        mock_catalog = Mock()
        mock_store.course_catalog = mock_catalog
        mock_catalog.get.side_effect = Exception("Database error")

        tool = CourseOutlineTool(mock_store)
        result = tool.execute(course_title="Test")

        assert "Error retrieving course outline" in result


class TestToolManager:
    """Test the ToolManager functionality"""

    def test_register_tool(self):
        """Test registering tools"""
        manager = ToolManager()
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"name": "test_tool"}

        manager.register_tool(mock_tool)

        assert "test_tool" in manager.tools
        assert manager.tools["test_tool"] == mock_tool

    def test_get_tool_definitions(self):
        """Test getting all tool definitions"""
        manager = ToolManager()

        mock_tool1 = Mock()
        mock_tool1.get_tool_definition.return_value = {"name": "tool1"}

        mock_tool2 = Mock()
        mock_tool2.get_tool_definition.return_value = {"name": "tool2"}

        manager.register_tool(mock_tool1)
        manager.register_tool(mock_tool2)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 2
        assert {"name": "tool1"} in definitions
        assert {"name": "tool2"} in definitions

    def test_execute_tool(self):
        """Test executing a registered tool"""
        manager = ToolManager()
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"name": "test_tool"}
        mock_tool.execute.return_value = "Tool result"

        manager.register_tool(mock_tool)

        result = manager.execute_tool("test_tool", query="test")

        assert result == "Tool result"
        mock_tool.execute.assert_called_once_with(query="test")

    def test_execute_nonexistent_tool(self):
        """Test executing a non-existent tool"""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent", query="test")

        assert "not found" in result

    def test_get_last_sources(self):
        """Test retrieving sources from tools"""
        manager = ToolManager()

        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"name": "search_tool"}
        mock_tool.last_sources = [{"text": "Source 1", "link": "https://link1.com"}]

        manager.register_tool(mock_tool)

        sources = manager.get_last_sources()

        assert len(sources) == 1
        assert sources[0]["text"] == "Source 1"

    def test_reset_sources(self):
        """Test resetting sources in all tools"""
        manager = ToolManager()

        mock_tool1 = Mock()
        mock_tool1.get_tool_definition.return_value = {"name": "tool1"}
        mock_tool1.last_sources = ["source1"]

        mock_tool2 = Mock()
        mock_tool2.get_tool_definition.return_value = {"name": "tool2"}
        mock_tool2.last_sources = ["source2"]

        manager.register_tool(mock_tool1)
        manager.register_tool(mock_tool2)

        manager.reset_sources()

        assert mock_tool1.last_sources == []
        assert mock_tool2.last_sources == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
