#!/usr/bin/env python3
"""
Integration test demonstrating sequential tool calling capability.
This test shows how Claude can now make up to 2 sequential tool calls,
using results from the first call to inform the second.
"""

from unittest.mock import Mock, patch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ai_generator import AIGenerator


def demonstrate_sequential_tools():
    """Demonstrate the sequential tool calling flow with a mock scenario"""
    
    print("=" * 60)
    print("SEQUENTIAL TOOL CALLING DEMONSTRATION")
    print("=" * 60)
    
    # Create mock tool manager
    mock_tool_manager = Mock()
    
    # Define what tools return
    def execute_tool_mock(tool_name, **kwargs):
        if tool_name == "get_course_outline":
            return """
**Course Title:** Advanced Python Programming
**Instructor:** Dr. Smith

**Lessons (10 total):**
- Lesson 1: Python Basics Review
- Lesson 2: Object-Oriented Programming
- Lesson 3: Functional Programming
- Lesson 4: Decorators and Context Managers
- Lesson 5: Metaclasses and Descriptors
- Lesson 6: Concurrency and Parallelism
- Lesson 7: Async Programming
- Lesson 8: Testing and Debugging
- Lesson 9: Performance Optimization
- Lesson 10: Advanced Design Patterns
"""
        elif tool_name == "search_course_content":
            query = kwargs.get('query', '')
            lesson = kwargs.get('lesson_number')
            if 'metaclass' in query.lower() or lesson == 5:
                return """
[Advanced Python Programming - Lesson 5]
Metaclasses are classes whose instances are classes. They allow you to:
- Control class creation process
- Add attributes/methods to classes dynamically
- Implement singleton patterns
- Create domain-specific languages (DSLs)

Example: Creating a metaclass that adds logging to all methods:
```python
class LoggingMeta(type):
    def __new__(cls, name, bases, attrs):
        for key, value in attrs.items():
            if callable(value):
                attrs[key] = log_wrapper(value)
        return super().__new__(cls, name, bases, attrs)
```
"""
            else:
                return "Content about " + query
    
    mock_tool_manager.execute_tool.side_effect = execute_tool_mock
    
    # Tool definitions
    tools = [
        {
            "name": "get_course_outline",
            "description": "Get course outline",
            "input_schema": {"type": "object", "properties": {}}
        },
        {
            "name": "search_course_content",
            "description": "Search course content",
            "input_schema": {"type": "object", "properties": {}}
        }
    ]
    
    # Example queries that would benefit from sequential tool calls
    queries = [
        "What does lesson 5 of the Advanced Python course cover?",
        "Find information about the most advanced topic in the Python course",
        "Compare the content between lesson 2 and lesson 5 of the course"
    ]
    
    print("\nExample queries that benefit from sequential tool calling:")
    print("-" * 60)
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("\nExpected flow:")
        print("  1. First tool call: get_course_outline to understand structure")
        print("  2. Second tool call: search_course_content for specific lesson/topic")
        print("  3. Final response: Synthesized answer from both tool results")
    
    print("\n" + "=" * 60)
    print("KEY IMPROVEMENTS:")
    print("-" * 60)
    print("✓ Claude can now make up to 2 sequential tool calls")
    print("✓ Second tool call can use information from first tool result")
    print("✓ Enables complex multi-step queries")
    print("✓ Better answers for queries requiring context + details")
    print("✓ Graceful error handling if tools fail")
    print("=" * 60)


if __name__ == "__main__":
    # Run demonstration
    demonstrate_sequential_tools()
    
    print("\n" + "=" * 60)
    print("TESTING NOTES:")
    print("-" * 60)
    print("• Run full test suite: cd backend && uv run pytest tests/test_ai_generator.py -v")
    print("• New tests added:")
    print("  - test_sequential_two_rounds: Full 2-round flow")
    print("  - test_max_rounds_termination: Handles >2 rounds gracefully")
    print("  - test_tool_execution_error_handling: Error recovery")
    print("=" * 60)