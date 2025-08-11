from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
1. **get_course_outline**: Use for course structure queries
   - Course syllabus or overview requests
   - Lesson listing or course content structure
   - Questions about what topics a course covers
   - Returns: Course title, link, instructor, and complete lesson list with links

2. **search_course_content**: Use for specific content queries
   - Detailed information about topics within courses
   - Specific concepts, definitions, or explanations
   - Content from particular lessons
   - Returns: Relevant content excerpts with context

Tool Usage Guidelines:
- **Sequential tool usage**: You can use tools up to 2 times to gather comprehensive information
- **Outline queries**: Use get_course_outline for structure, syllabus, or lesson lists
- **Content queries**: Use search_course_content for detailed topic information
- **Multi-step research**: First tool call can inform your second tool call for better results
- **Smart combinations**: Get course outline first, then search specific lessons based on what you find
- Synthesize all tool results into accurate, fact-based responses
- If tools yield no results, state this clearly

Examples of multi-tool usage:
- "What advanced topics are in the Python course?" → Get outline → Search for advanced content in later lessons
- "Compare topic X between courses A and B" → Search course A → Search course B with same query
- "Find details about lesson 4 of course X" → Get outline to see lesson 4 title → Search for that specific content

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course-specific questions**: Use appropriate tools (up to 2 calls), then answer
- **Complex queries**: Break down into multiple tool calls when beneficial for completeness
- **No meta-commentary**: Provide direct answers only — no tool explanations
- Format outline information clearly with course details and lesson structure

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Well-formatted** - Use markdown for better readability when showing outlines
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)

        # Return direct response
        return response.content[0].text

    def _execute_tool_round(
        self, response, tool_manager
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute all tools in a response and return results.

        Args:
            response: API response potentially containing tool use blocks
            tool_manager: Manager to execute tools

        Returns:
            List of tool results or None if no tools were executed
        """
        tool_results = []

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )
                except Exception as e:
                    # Return error result instead of failing completely
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Tool execution failed: {str(e)}",
                        }
                    )

        return tool_results if tool_results else None

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Handle execution of tool calls with support for up to 2 sequential rounds.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Initialize conversation state
        messages = base_params["messages"].copy()
        current_response = initial_response
        max_rounds = 2
        round_count = 0

        # Process tool calls in sequential rounds
        while round_count < max_rounds and current_response.stop_reason == "tool_use":
            round_count += 1

            # Add Claude's response to message history
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute tools for this round
            tool_results = self._execute_tool_round(current_response, tool_manager)

            if not tool_results:
                # No tools executed or error occurred
                return "I encountered an issue while using the available tools."

            # Add tool results to conversation
            messages.append({"role": "user", "content": tool_results})

            # Check if we've reached max rounds
            if round_count >= max_rounds:
                # Max rounds reached, make final call without tools
                break

            # Make next API call with tools still available for potential second round
            next_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"],
                "tools": base_params.get("tools", []),  # Keep tools available
                "tool_choice": {"type": "auto"},
            }

            try:
                current_response = self.client.messages.create(**next_params)

                # If Claude responds with text only (no more tools), return that response
                if current_response.stop_reason != "tool_use":
                    return current_response.content[0].text

            except Exception as e:
                return f"Error during tool execution round {round_count}: {str(e)}"

        # After loop: either max rounds reached or Claude stopped using tools
        # Make final call without tools to get text response
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
            # No tools parameter - forcing text response
        }

        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
