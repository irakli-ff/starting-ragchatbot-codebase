import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_generator import AIGenerator


class TestAIGenerator:
    """Test the AIGenerator functionality"""

    def test_initialization(self):
        """Test AIGenerator initialization"""
        generator = AIGenerator(api_key="test-key", model="test-model")

        assert generator.client is not None
        assert generator.model == "test-model"
        assert generator.base_params["model"] == "test-model"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_without_tools(self, mock_anthropic_class):
        """Test generating response without tools"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.stop_reason = "end"
        mock_response.content = [Mock(text="This is a simple answer")]
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="test-model")

        # Test without tools
        result = generator.generate_response(
            query="What is Python?",
            conversation_history=None,
            tools=None,
            tool_manager=None,
        )

        assert result == "This is a simple answer"
        mock_client.messages.create.assert_called_once()

        # Check the call arguments
        call_args = mock_client.messages.create.call_args[1]
        assert call_args["model"] == "test-model"
        assert call_args["messages"][0]["content"] == "What is Python?"
        assert "tools" not in call_args

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_with_conversation_history(self, mock_anthropic_class):
        """Test generating response with conversation history"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.stop_reason = "end"
        mock_response.content = [Mock(text="Answer with context")]
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="test-model")

        history = "User: Previous question\nAssistant: Previous answer"
        result = generator.generate_response(
            query="Follow-up question", conversation_history=history
        )

        assert result == "Answer with context"

        call_args = mock_client.messages.create.call_args[1]
        assert "Previous conversation:" in call_args["system"]
        assert history in call_args["system"]

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_with_tools(self, mock_anthropic_class):
        """Test generating response with tools available"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.stop_reason = "end"
        mock_response.content = [Mock(text="Answer using tools")]
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="test-model")

        tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]

        result = generator.generate_response(
            query="Search for Python basics", tools=tools
        )

        assert result == "Answer using tools"

        call_args = mock_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert call_args["tools"] == tools
        assert call_args["tool_choice"] == {"type": "auto"}

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_execution_flow(self, mock_anthropic_class):
        """Test the single-round tool execution flow when AI decides to use tools"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First response - AI decides to use tool
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "Python basics"}
        mock_tool_content.id = "tool_123"
        mock_tool_response.content = [mock_tool_content]

        # Second response - AI decides no more tools needed
        mock_second_response = Mock()
        mock_second_response.stop_reason = "end"
        mock_second_response.content = [
            Mock(text="Here's what I found about Python basics...")
        ]

        # Third response would be final, but shouldn't be called since second has stop_reason="end"
        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_second_response,
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = (
            "Python is a programming language..."
        )

        generator = AIGenerator(api_key="test-key", model="test-model")

        tools = [{"name": "search_course_content"}]
        result = generator.generate_response(
            query="Tell me about Python basics",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        assert result == "Here's what I found about Python basics..."

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="Python basics"
        )

        # Verify exactly 2 API calls were made (initial + one with tools still available)
        assert mock_client.messages.create.call_count == 2

        # Check second call still has tools available (key change!)
        second_call_args = mock_client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_args  # Tools should still be available
        assert second_call_args["tool_choice"] == {"type": "auto"}

        # Check messages structure
        messages = second_call_args["messages"]
        assert (
            len(messages) == 3
        )  # Original user + assistant with tool + user with results
        assert messages[1]["role"] == "assistant"  # AI's tool use
        assert messages[2]["role"] == "user"  # Tool results
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert (
            messages[2]["content"][0]["content"]
            == "Python is a programming language..."
        )

    @patch("ai_generator.anthropic.Anthropic")
    def test_multiple_tool_calls(self, mock_anthropic_class):
        """Test handling multiple tool calls in one response"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First response - AI decides to use multiple tools
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"

        tool1 = Mock()
        tool1.type = "tool_use"
        tool1.name = "search_course_content"
        tool1.input = {"query": "Python"}
        tool1.id = "tool_1"

        tool2 = Mock()
        tool2.type = "tool_use"
        tool2.name = "get_course_outline"
        tool2.input = {"course_title": "Python Course"}
        tool2.id = "tool_2"

        mock_tool_response.content = [tool1, tool2]

        # Second response - Final answer
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end"
        mock_final_response.content = [Mock(text="Combined results")]

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        generator = AIGenerator(api_key="test-key", model="test-model")

        result = generator.generate_response(
            query="Complex query",
            tools=[{"name": "search_course_content"}, {"name": "get_course_outline"}],
            tool_manager=mock_tool_manager,
        )

        assert result == "Combined results"
        assert mock_tool_manager.execute_tool.call_count == 2

    def test_system_prompt_content(self):
        """Test that system prompt contains expected content"""
        generator = AIGenerator(api_key="test-key", model="test-model")

        assert "get_course_outline" in generator.SYSTEM_PROMPT
        assert "search_course_content" in generator.SYSTEM_PROMPT
        assert "Tool Usage Guidelines" in generator.SYSTEM_PROMPT
        assert "Sequential tool usage" in generator.SYSTEM_PROMPT
        assert "up to 2 times" in generator.SYSTEM_PROMPT
        assert "Brief, Concise and focused" in generator.SYSTEM_PROMPT

    @patch("ai_generator.anthropic.Anthropic")
    def test_sequential_two_rounds(self, mock_anthropic_class):
        """Test full two-round sequential tool calling flow"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Round 1: AI uses get_course_outline tool
        mock_round1_response = Mock()
        mock_round1_response.stop_reason = "tool_use"
        mock_tool1 = Mock()
        mock_tool1.type = "tool_use"
        mock_tool1.name = "get_course_outline"
        mock_tool1.input = {"course_title": "Python Course"}
        mock_tool1.id = "tool_1"
        mock_round1_response.content = [mock_tool1]

        # Round 2: After seeing outline, AI uses search_course_content
        mock_round2_response = Mock()
        mock_round2_response.stop_reason = "tool_use"
        mock_tool2 = Mock()
        mock_tool2.type = "tool_use"
        mock_tool2.name = "search_course_content"
        mock_tool2.input = {"query": "advanced topics", "lesson_number": 10}
        mock_tool2.id = "tool_2"
        mock_round2_response.content = [mock_tool2]

        # Final response after 2 rounds
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end"
        mock_final_response.content = [
            Mock(text="Based on the course outline and search results...")
        ]

        mock_client.messages.create.side_effect = [
            mock_round1_response,  # Initial call
            mock_round2_response,  # After first tool
            mock_final_response,  # After second tool (max rounds reached)
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course Title: Python\nLessons:\n- Lesson 1: Basics\n- Lesson 10: Advanced Topics",
            "Advanced topics include decorators, metaclasses, and async programming...",
        ]

        generator = AIGenerator(api_key="test-key", model="test-model")

        tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]

        result = generator.generate_response(
            query="What advanced topics are covered in the Python course?",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        assert result == "Based on the course outline and search results..."

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline", course_title="Python Course"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content", query="advanced topics", lesson_number=10
        )

        # Verify 3 API calls were made
        assert mock_client.messages.create.call_count == 3

        # Check that tools were available in round 2
        second_call_args = mock_client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_args

        # Check final call has no tools (max rounds reached)
        final_call_args = mock_client.messages.create.call_args_list[2][1]
        assert "tools" not in final_call_args

    @patch("ai_generator.anthropic.Anthropic")
    def test_max_rounds_termination(self, mock_anthropic_class):
        """Test behavior when Claude wants more than 2 rounds"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Create responses that always want to use tools
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool = Mock()
        mock_tool.type = "tool_use"
        mock_tool.name = "search_course_content"
        mock_tool.input = {"query": "test"}
        mock_tool.id = "tool_id"
        mock_tool_response.content = [mock_tool]

        # Final response when no tools available
        mock_final = Mock()
        mock_final.stop_reason = "end"
        mock_final.content = [Mock(text="Final answer after 2 rounds")]

        mock_client.messages.create.side_effect = [
            mock_tool_response,  # Round 1
            mock_tool_response,  # Round 2 (still wants tools)
            mock_final,  # Forced final without tools
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        generator = AIGenerator(api_key="test-key", model="test-model")

        result = generator.generate_response(
            query="Complex query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        assert result == "Final answer after 2 rounds"

        # Verify exactly 2 tool executions
        assert mock_tool_manager.execute_tool.call_count == 2

        # Verify 3 API calls total
        assert mock_client.messages.create.call_count == 3

        # Verify last call has no tools
        final_call_args = mock_client.messages.create.call_args_list[2][1]
        assert "tools" not in final_call_args

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_execution_error_handling(self, mock_anthropic_class):
        """Test graceful handling of tool execution errors"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # AI decides to use tool
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test"}
        mock_tool_content.id = "tool_123"
        mock_tool_response.content = [mock_tool_content]

        # Response after seeing error
        mock_final = Mock()
        mock_final.stop_reason = "end"
        mock_final.content = [
            Mock(text="I encountered an error but here's what I can tell you...")
        ]

        mock_client.messages.create.side_effect = [mock_tool_response, mock_final]

        # Mock tool manager that throws error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception(
            "Database connection failed"
        )

        generator = AIGenerator(api_key="test-key", model="test-model")

        result = generator.generate_response(
            query="Search for something",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        assert result == "I encountered an error but here's what I can tell you..."

        # Verify tool execution was attempted
        mock_tool_manager.execute_tool.assert_called_once()

        # Verify error was handled and passed to Claude
        second_call_args = mock_client.messages.create.call_args_list[1][1]
        messages = second_call_args["messages"]
        tool_result = messages[-1]["content"][0]
        assert "Tool execution failed" in tool_result["content"]
        assert "Database connection failed" in tool_result["content"]

    @patch("ai_generator.anthropic.Anthropic")
    def test_error_handling_in_api_call(self, mock_anthropic_class):
        """Test that API errors are properly propagated"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_client.messages.create.side_effect = Exception("API Error")

        generator = AIGenerator(api_key="test-key", model="test-model")

        with pytest.raises(Exception) as exc_info:
            generator.generate_response(query="Test query")

        assert "API Error" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
