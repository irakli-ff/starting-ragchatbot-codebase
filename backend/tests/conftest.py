import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any
import asyncio

import pytest

# Add parent directory to path to import backend modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from models import Course, CourseChunk, Lesson
from vector_store import SearchResults, VectorStore


@pytest.fixture
def mock_config():
    """Create a test configuration"""
    config = Config()
    config.ANTHROPIC_API_KEY = "test-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    return config


@pytest.fixture
def temp_chroma_db():
    """Create a temporary ChromaDB instance for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    course = Course(
        title="Test Course on Python Programming",
        course_link="https://example.com/course",
        instructor="John Doe",
    )
    course.lessons = [
        Lesson(
            lesson_number=0,
            title="Introduction to Python",
            lesson_link="https://example.com/lesson0",
        ),
        Lesson(
            lesson_number=1,
            title="Variables and Data Types",
            lesson_link="https://example.com/lesson1",
        ),
        Lesson(lesson_number=2, title="Control Flow", lesson_link=None),
    ]
    return course


@pytest.fixture
def sample_chunks(sample_course):
    """Create sample course chunks for testing"""
    chunks = []
    chunk_index = 0

    # Map lesson content for testing
    lesson_content = {
        0: "Welcome to Python programming. Python is a versatile language.",
        1: "In Python, variables store data. Common data types include strings, integers, and lists.",
        2: "Control flow in Python uses if statements, for loops, and while loops to control program execution.",
    }

    for lesson in sample_course.lessons:
        # Create chunks for each lesson
        chunk = CourseChunk(
            course_title=sample_course.title,
            lesson_number=lesson.lesson_number,
            content=lesson_content.get(lesson.lesson_number, "Default content"),
            chunk_index=chunk_index,
        )
        chunks.append(chunk)
        chunk_index += 1

    return chunks


@pytest.fixture
def sample_search_results():
    """Create sample search results for testing"""
    return SearchResults(
        documents=[
            "Python is a versatile programming language.",
            "Variables in Python can store different data types.",
        ],
        metadata=[
            {"course_title": "Test Course on Python Programming", "lesson_number": 0},
            {"course_title": "Test Course on Python Programming", "lesson_number": 1},
        ],
        distances=[0.1, 0.2],
    )


@pytest.fixture
def sample_course_doc(tmp_path):
    """Create a sample course document file for testing"""
    doc_content = """Course Title: Building Towards Computer Use with Anthropic
Course Link: https://www.deeplearning.ai/short-courses/building-toward-computer-use-with-anthropic/
Course Instructor: Colt Steele

Lesson 0: Introduction
Lesson Link: https://learn.deeplearning.ai/courses/lesson0
Welcome to Building Toward Computer Use with Anthropic. In this course, you will learn about computer use capabilities.

Lesson 1: Getting Started
Lesson Link: https://learn.deeplearning.ai/courses/lesson1
Let's begin by understanding the basics of how LLMs can control computers.

Lesson 2: Advanced Techniques
This lesson covers advanced techniques for computer use without a specific link.
"""

    doc_path = tmp_path / "test_course.txt"
    doc_path.write_text(doc_content)
    return str(doc_path)


@pytest.fixture
def mock_anthropic_response():
    """Mock response from Anthropic API"""

    class MockContent:
        def __init__(self, text=None, type="text", name=None, input=None, id=None):
            self.text = text
            self.type = type
            self.name = name
            self.input = input
            self.id = id

    class MockResponse:
        def __init__(
            self, content_text="Test response", stop_reason="end", tool_use=False
        ):
            if tool_use:
                self.stop_reason = "tool_use"
                self.content = [
                    MockContent(
                        type="tool_use",
                        name="search_course_content",
                        input={"query": "Python basics"},
                        id="tool_123",
                    )
                ]
            else:
                self.stop_reason = stop_reason
                self.content = [MockContent(text=content_text)]

    return MockResponse


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing"""
    mock_client = Mock()
    mock_messages = Mock()
    mock_client.messages = mock_messages
    
    # Default response
    mock_response = Mock()
    mock_response.content = [Mock(text="Mocked AI response")]
    mock_response.stop_reason = "end"
    mock_messages.create.return_value = mock_response
    
    return mock_client


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    mock_store = Mock(spec=VectorStore)
    
    # Default search results
    mock_store.search.return_value = SearchResults(
        documents=["Test document content"],
        metadata=[{"course_title": "Test Course", "lesson_number": 0}],
        distances=[0.1]
    )
    
    mock_store.search_courses.return_value = SearchResults(
        documents=["Test Course on Python Programming"],
        metadata=[{"course_title": "Test Course on Python Programming"}],
        distances=[0.05]
    )
    
    mock_store.get_all_courses.return_value = [
        {
            "title": "Test Course on Python Programming",
            "instructor": "John Doe",
            "lessons": 5
        }
    ]
    
    return mock_store


@pytest.fixture
def mock_session_manager():
    """Mock session manager for testing"""
    from backend.session_manager import SessionManager
    
    mock_manager = Mock(spec=SessionManager)
    mock_manager.get_or_create_session.return_value = "test-session-id"
    mock_manager.get_conversation_history.return_value = []
    mock_manager.add_exchange.return_value = None
    
    return mock_manager


@pytest.fixture
def mock_rag_system():
    """Mock RAG system for testing"""
    mock_rag = Mock()
    
    # Default process_query response
    mock_rag.process_query.return_value = {
        "answer": "This is a test response from the RAG system.",
        "sources": [
            {"text": "Lesson 1: Introduction", "link": "https://example.com/lesson1"}
        ],
        "session_id": "test-session-123"
    }
    
    # Default get_all_courses response
    mock_rag.get_all_courses.return_value = [
        {
            "title": "Python Programming",
            "instructor": "John Doe",
            "lessons": 10,
            "link": "https://example.com/python"
        }
    ]
    
    # Default get_stats response
    mock_rag.get_stats.return_value = {
        "total_courses": 2,
        "total_lessons": 15,
        "total_chunks": 120,
        "embedding_model": "all-MiniLM-L6-v2"
    }
    
    return mock_rag


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory with test data files"""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    
    # Create sample course files
    course1 = test_dir / "python_basics.txt"
    course1.write_text("""Course Title: Python Basics
Course Link: https://example.com/python
Course Instructor: John Doe

Lesson 0: Introduction
Welcome to Python programming.

Lesson 1: Variables
Learn about variables and data types.
""")
    
    course2 = test_dir / "data_science.txt"
    course2.write_text("""Course Title: Data Science with Python
Course Link: https://example.com/datascience
Course Instructor: Jane Smith

Lesson 0: Overview
Introduction to data science concepts.

Lesson 1: NumPy Basics
Working with NumPy arrays.
""")
    
    return test_dir


@pytest.fixture
def mock_tool_manager():
    """Mock tool manager for testing"""
    from backend.search_tools import ToolManager
    
    mock_manager = Mock(spec=ToolManager)
    mock_manager.execute_tool.return_value = {
        "results": [
            {
                "content": "Python is a high-level programming language",
                "metadata": {"course_title": "Python Basics", "lesson_number": 0}
            }
        ]
    }
    
    return mock_manager


@pytest.fixture
def mock_env_variables(monkeypatch):
    """Set up mock environment variables"""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    monkeypatch.setenv("CHROMA_DB_PATH", "./test_chroma_db")
    monkeypatch.setenv("DOCS_PATH", "./test_docs")
    return {
        "ANTHROPIC_API_KEY": "test-api-key",
        "CHROMA_DB_PATH": "./test_chroma_db",
        "DOCS_PATH": "./test_docs"
    }


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests"""
    # This ensures clean state between tests
    yield
    # Clean up any singleton instances if needed


@pytest.fixture
def async_mock():
    """Helper to create async mock objects"""
    def _create_async_mock(return_value=None):
        async def async_func(*args, **kwargs):
            return return_value
        
        mock = Mock()
        mock.side_effect = async_func
        return mock
    
    return _create_async_mock


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client"""
    mock_client = Mock()
    mock_collection = Mock()
    
    # Mock collection methods
    mock_collection.query.return_value = {
        "documents": [["Test document"]],
        "metadatas": [[{"course_title": "Test Course"}]],
        "distances": [[0.1]]
    }
    
    mock_collection.add.return_value = None
    mock_collection.delete.return_value = None
    mock_collection.count.return_value = 10
    
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client.list_collections.return_value = [mock_collection]
    
    return mock_client


@pytest.fixture
def sample_query_responses():
    """Sample query responses for testing"""
    return {
        "direct_answer": {
            "answer": "Python is a high-level, interpreted programming language.",
            "sources": [],
            "session_id": "session-001"
        },
        "with_search": {
            "answer": "Based on the course materials, Python uses dynamic typing where variable types are determined at runtime.",
            "sources": [
                {"text": "Variables and Data Types", "link": "https://example.com/lesson1"},
                {"text": "Python Type System", "link": "https://example.com/lesson2"}
            ],
            "session_id": "session-002"
        },
        "course_outline": {
            "answer": "Here's the course outline for Python Basics:\n\n1. Introduction\n2. Variables and Data Types\n3. Control Flow",
            "sources": [
                {"text": "Python Basics Course", "link": "https://example.com/python"}
            ],
            "session_id": "session-003"
        }
    }


@pytest.fixture
def performance_timer():
    """Helper fixture for performance testing"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()
