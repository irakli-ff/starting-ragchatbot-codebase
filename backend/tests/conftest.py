import os
import sys
import tempfile
from pathlib import Path

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
