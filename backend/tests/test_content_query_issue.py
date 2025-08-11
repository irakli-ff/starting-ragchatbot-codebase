"""
Integration test to diagnose why content queries are returning "query failed"
This test simulates the actual system flow to identify the failure point.
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_system import RAGSystem
from config import Config
from models import Course, Lesson, CourseChunk
from document_processor import DocumentProcessor
from vector_store import VectorStore
from ai_generator import AIGenerator
from search_tools import CourseSearchTool, ToolManager


class TestContentQueryIssue:
    """Test suite to diagnose content query failures"""
    
    def test_document_loading_from_docs_folder(self, temp_chroma_db):
        """Test if documents are properly loaded from docs folder"""
        # Create config with temp database
        config = Config()
        config.CHROMA_PATH = temp_chroma_db
        config.ANTHROPIC_API_KEY = "test-key"
        
        # Create processor and test parsing a real course document
        processor = DocumentProcessor(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        
        # Check if docs folder exists and has files
        docs_path = "../docs"
        abs_docs_path = os.path.abspath(docs_path)
        print(f"\n[TEST] Checking docs folder: {abs_docs_path}")
        
        if os.path.exists(docs_path):
            files = os.listdir(docs_path)
            print(f"[TEST] Found {len(files)} files: {files}")
            
            # Try to process the first course document
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(docs_path, file)
                    print(f"[TEST] Processing: {file}")
                    
                    try:
                        course, chunks = processor.process_course_document(file_path)
                        print(f"[TEST] Successfully parsed: {course.title}")
                        print(f"[TEST] Course has {len(course.lessons)} lessons")
                        print(f"[TEST] Generated {len(chunks)} chunks")
                        
                        # Check chunk content
                        if chunks:
                            print(f"[TEST] First chunk preview: {chunks[0].content[:100]}...")
                        
                        assert course is not None
                        assert len(chunks) > 0
                        return  # Success
                        
                    except Exception as e:
                        print(f"[TEST] Error processing {file}: {e}")
                        raise
        else:
            pytest.skip("Docs folder not found")
    
    def test_vector_store_population(self, temp_chroma_db):
        """Test if vector store is properly populated with documents"""
        config = Config()
        config.CHROMA_PATH = temp_chroma_db
        config.ANTHROPIC_API_KEY = "test-key"
        
        # Create RAG system
        rag_system = RAGSystem(config)
        
        # Add documents from docs folder
        docs_path = "../docs"
        if os.path.exists(docs_path):
            courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=True)
            print(f"\n[TEST] Added {courses} courses with {chunks} chunks")
            
            # Check if data is in vector store
            analytics = rag_system.get_course_analytics()
            print(f"[TEST] Vector store contains {analytics['total_courses']} courses")
            print(f"[TEST] Course titles: {analytics['course_titles']}")
            
            assert analytics['total_courses'] > 0
            
            # Test direct search in vector store
            results = rag_system.vector_store.search("Python", limit=5)
            print(f"[TEST] Direct search for 'Python' returned {len(results.documents)} results")
            
            if not results.is_empty():
                print(f"[TEST] First result: {results.documents[0][:100]}...")
        else:
            pytest.skip("Docs folder not found")
    
    def test_search_tool_execution(self, temp_chroma_db):
        """Test if CourseSearchTool properly executes searches"""
        config = Config()
        config.CHROMA_PATH = temp_chroma_db
        config.ANTHROPIC_API_KEY = "test-key"
        
        # Create vector store and add some test data
        store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL)
        
        # Add test course
        course = Course(
            title="Test Python Course",
            instructor="Test Instructor"
        )
        course.lessons = [
            Lesson(lesson_number=0, title="Introduction")
        ]
        
        chunks = [
            CourseChunk(
                course_title="Test Python Course",
                lesson_number=0,
                content="Python is a high-level programming language.",
                chunk_index=0
            )
        ]
        
        store.add_course_metadata(course)
        store.add_course_content(chunks)
        
        # Create and test search tool
        search_tool = CourseSearchTool(store)
        
        print("\n[TEST] Testing CourseSearchTool execution")
        result = search_tool.execute(query="Python programming")
        print(f"[TEST] Search result: {result[:200]}...")
        
        assert result is not None
        assert "No relevant content found" not in result
        
        # Check sources are tracked
        print(f"[TEST] Sources tracked: {search_tool.last_sources}")
        assert len(search_tool.last_sources) > 0
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_ai_tool_calling_decision(self, mock_anthropic_class, temp_chroma_db):
        """Test if AI correctly decides to use tools for content queries"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock response that uses tool
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "Python basics"}
        mock_tool_content.id = "tool_123"
        mock_tool_response.content = [mock_tool_content]
        
        # Mock final response
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end"
        mock_final_response.content = [Mock(text="Here's information about Python basics...")]
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Create AI generator
        generator = AIGenerator(api_key="test-key", model="test-model")
        
        # Create tool manager with search tool
        tool_manager = ToolManager()
        mock_search_tool = Mock()
        mock_search_tool.get_tool_definition.return_value = {
            "name": "search_course_content",
            "description": "Search course content"
        }
        mock_search_tool.execute.return_value = "Python is a programming language..."
        tool_manager.register_tool(mock_search_tool)
        
        # Test content query
        print("\n[TEST] Testing AI tool calling for content query")
        response = generator.generate_response(
            query="What are Python basics?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        print(f"[TEST] AI response: {response}")
        
        # Verify tool was called
        assert mock_search_tool.execute.called
        assert "Python basics" in response
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_full_query_flow_simulation(self, mock_anthropic_class, temp_chroma_db):
        """Simulate the full query flow to identify failure point"""
        print("\n[TEST] === FULL QUERY FLOW SIMULATION ===")
        
        # Setup config
        config = Config()
        config.CHROMA_PATH = temp_chroma_db
        config.ANTHROPIC_API_KEY = "test-key"
        
        # Mock Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Create tool use response
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "Python variables"}
        mock_tool_content.id = "tool_123"
        mock_tool_response.content = [mock_tool_content]
        
        # Create final response
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end"
        mock_final_response.content = [Mock(text="Variables in Python are used to store data.")]
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Create RAG system
        print("[TEST] 1. Creating RAG system")
        rag_system = RAGSystem(config)
        
        # Add test data
        print("[TEST] 2. Adding test course data")
        course = Course(
            title="Python Basics",
            instructor="Test"
        )
        course.lessons = [Lesson(lesson_number=0, title="Variables")]
        
        chunks = [
            CourseChunk(
                course_title="Python Basics",
                lesson_number=0,
                content="Variables in Python are containers for storing data values.",
                chunk_index=0
            )
        ]
        
        rag_system.vector_store.add_course_metadata(course)
        rag_system.vector_store.add_course_content(chunks)
        
        # Perform query
        print("[TEST] 3. Executing query: 'What are Python variables?'")
        try:
            response, sources = rag_system.query("What are Python variables?")
            print(f"[TEST] Response: {response}")
            print(f"[TEST] Sources: {sources}")
            
            assert response is not None
            assert "Variables" in response or "variables" in response
            
        except Exception as e:
            print(f"[TEST] ERROR during query: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def test_api_key_configuration(self):
        """Test if API key is properly configured"""
        config = Config()
        
        print(f"\n[TEST] Checking API key configuration")
        print(f"[TEST] API key present: {bool(config.ANTHROPIC_API_KEY)}")
        print(f"[TEST] API key length: {len(config.ANTHROPIC_API_KEY) if config.ANTHROPIC_API_KEY else 0}")
        
        if not config.ANTHROPIC_API_KEY:
            print("[TEST] WARNING: No API key found in environment")
    
    def test_error_handling_in_query(self, temp_chroma_db):
        """Test how errors are handled in the query flow"""
        config = Config()
        config.CHROMA_PATH = temp_chroma_db
        config.ANTHROPIC_API_KEY = "test-key"
        
        # Create RAG system with mocked AI that fails
        with patch('rag_system.AIGenerator') as mock_ai_class:
            mock_ai = Mock()
            mock_ai_class.return_value = mock_ai
            mock_ai.generate_response.side_effect = Exception("API Error")
            
            rag_system = RAGSystem(config)
            
            print("\n[TEST] Testing error handling in query")
            try:
                response, sources = rag_system.query("Test query")
                print(f"[TEST] Unexpected success: {response}")
            except Exception as e:
                print(f"[TEST] Expected error caught: {e}")
                assert "API Error" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])