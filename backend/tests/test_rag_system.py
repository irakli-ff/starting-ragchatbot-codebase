import pytest
from unittest.mock import Mock, MagicMock, patch
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_system import RAGSystem
from config import Config
from models import Course, Lesson, CourseChunk


class TestRAGSystem:
    """Test the RAGSystem end-to-end functionality"""
    
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_initialization(self, mock_doc_processor, mock_vector_store, mock_ai_generator, mock_config):
        """Test RAGSystem initialization"""
        rag_system = RAGSystem(mock_config)
        
        assert rag_system.config == mock_config
        assert rag_system.document_processor is not None
        assert rag_system.vector_store is not None
        assert rag_system.ai_generator is not None
        assert rag_system.session_manager is not None
        assert rag_system.tool_manager is not None
        assert rag_system.search_tool is not None
        assert rag_system.outline_tool is not None
        
        # Verify tools are registered
        assert len(rag_system.tool_manager.tools) == 2
    
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_add_course_document_success(self, mock_doc_processor_class, mock_vector_store_class, 
                                         mock_ai_generator_class, mock_config, sample_course, sample_chunks):
        """Test successfully adding a course document"""
        # Setup mocks
        mock_doc_processor = Mock()
        mock_doc_processor_class.return_value = mock_doc_processor
        mock_doc_processor.process_course_document.return_value = (sample_course, sample_chunks)
        
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        
        rag_system = RAGSystem(mock_config)
        
        # Add course document
        course, num_chunks = rag_system.add_course_document("/path/to/course.txt")
        
        assert course == sample_course
        assert num_chunks == len(sample_chunks)
        
        # Verify methods were called
        mock_doc_processor.process_course_document.assert_called_once_with("/path/to/course.txt")
        mock_vector_store.add_course_metadata.assert_called_once_with(sample_course)
        mock_vector_store.add_course_content.assert_called_once_with(sample_chunks)
    
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_add_course_document_error(self, mock_doc_processor_class, mock_vector_store_class,
                                       mock_ai_generator_class, mock_config):
        """Test error handling when adding course document fails"""
        mock_doc_processor = Mock()
        mock_doc_processor_class.return_value = mock_doc_processor
        mock_doc_processor.process_course_document.side_effect = Exception("Parse error")
        
        rag_system = RAGSystem(mock_config)
        
        course, num_chunks = rag_system.add_course_document("/path/to/bad.txt")
        
        assert course is None
        assert num_chunks == 0
    
    @patch('rag_system.os.path.exists')
    @patch('rag_system.os.listdir')
    @patch('rag_system.os.path.isfile')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_add_course_folder(self, mock_doc_processor_class, mock_vector_store_class,
                               mock_ai_generator_class, mock_isfile, mock_listdir, 
                               mock_exists, mock_config, sample_course, sample_chunks):
        """Test adding all documents from a folder"""
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt", "course2.pdf", "readme.md"]
        mock_isfile.side_effect = [True, True, True]
        
        mock_doc_processor = Mock()
        mock_doc_processor_class.return_value = mock_doc_processor
        mock_doc_processor.process_course_document.return_value = (sample_course, sample_chunks)
        
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        mock_vector_store.get_existing_course_titles.return_value = []
        
        rag_system = RAGSystem(mock_config)
        
        total_courses, total_chunks = rag_system.add_course_folder("/docs")
        
        # Should process 2 files (txt and pdf, not md)
        assert total_courses == 2
        assert total_chunks == 2 * len(sample_chunks)
    
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_query_without_session(self, mock_doc_processor_class, mock_vector_store_class,
                                  mock_ai_generator_class, mock_config):
        """Test querying without a session ID"""
        # Setup mocks
        mock_ai_generator = Mock()
        mock_ai_generator_class.return_value = mock_ai_generator
        mock_ai_generator.generate_response.return_value = "Test response"
        
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        
        rag_system = RAGSystem(mock_config)
        
        response, sources = rag_system.query("What is Python?")
        
        assert response == "Test response"
        assert isinstance(sources, list)
        
        # Verify AI generator was called with tools
        mock_ai_generator.generate_response.assert_called_once()
        call_args = mock_ai_generator.generate_response.call_args[1]
        assert "tools" in call_args
        assert call_args["tool_manager"] is not None
    
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_query_with_session(self, mock_doc_processor_class, mock_vector_store_class,
                                mock_ai_generator_class, mock_config):
        """Test querying with a session ID for conversation context"""
        # Setup mocks
        mock_ai_generator = Mock()
        mock_ai_generator_class.return_value = mock_ai_generator
        mock_ai_generator.generate_response.return_value = "Response with context"
        
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        
        rag_system = RAGSystem(mock_config)
        
        # Create a session
        session_id = rag_system.session_manager.create_session()
        
        # Add some history
        rag_system.session_manager.add_exchange(session_id, "Previous question", "Previous answer")
        
        response, sources = rag_system.query("Follow-up question", session_id)
        
        assert response == "Response with context"
        
        # Verify conversation history was passed
        call_args = mock_ai_generator.generate_response.call_args[1]
        assert call_args["conversation_history"] is not None
        assert "Previous question" in call_args["conversation_history"]
    
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_query_with_tool_execution(self, mock_doc_processor_class, mock_vector_store_class,
                                       mock_ai_generator_class, mock_config):
        """Test query that triggers tool execution"""
        # Setup mocks
        mock_ai_generator = Mock()
        mock_ai_generator_class.return_value = mock_ai_generator
        mock_ai_generator.generate_response.return_value = "Found information about Python"
        
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        
        rag_system = RAGSystem(mock_config)
        
        # Mock tool manager to simulate tool execution
        rag_system.tool_manager.get_last_sources = Mock(return_value=[
            {"text": "Python Course - Lesson 1", "link": "https://example.com/lesson1"}
        ])
        
        response, sources = rag_system.query("Search for Python basics")
        
        assert response == "Found information about Python"
        assert len(sources) == 1
        assert sources[0]["text"] == "Python Course - Lesson 1"
        
        # Verify sources were reset after retrieval
        rag_system.tool_manager.reset_sources.assert_called_once()
    
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_get_course_analytics(self, mock_doc_processor_class, mock_vector_store_class,
                                  mock_ai_generator_class, mock_config):
        """Test getting course analytics"""
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        mock_vector_store.get_course_count.return_value = 5
        mock_vector_store.get_existing_course_titles.return_value = [
            "Course 1", "Course 2", "Course 3", "Course 4", "Course 5"
        ]
        
        rag_system = RAGSystem(mock_config)
        
        analytics = rag_system.get_course_analytics()
        
        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course 1" in analytics["course_titles"]
    
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_query_types_differentiation(self, mock_doc_processor_class, mock_vector_store_class,
                                         mock_ai_generator_class, mock_config):
        """Test that different query types are handled appropriately"""
        mock_ai_generator = Mock()
        mock_ai_generator_class.return_value = mock_ai_generator
        
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        
        rag_system = RAGSystem(mock_config)
        
        # Test outline query
        mock_ai_generator.generate_response.return_value = "Course outline response"
        response, _ = rag_system.query("Show me the course outline for Python")
        assert "Course outline response" in response
        
        # Test content search query
        mock_ai_generator.generate_response.return_value = "Search results for variables"
        response, _ = rag_system.query("What are variables in Python?")
        assert "Search results for variables" in response
        
        # Test general knowledge query
        mock_ai_generator.generate_response.return_value = "General knowledge response"
        response, _ = rag_system.query("What is the capital of France?")
        assert "General knowledge response" in response
        
        # All queries should be passed with tools available
        assert mock_ai_generator.generate_response.call_count == 3
        for call in mock_ai_generator.generate_response.call_args_list:
            assert "tools" in call[1]
            assert len(call[1]["tools"]) == 2  # Both search and outline tools


class TestRAGSystemIntegration:
    """Integration tests with real components (but mocked API)"""
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_full_query_flow_with_mock_api(self, mock_anthropic_class, mock_config, temp_chroma_db):
        """Test full query flow with mocked Anthropic API"""
        # Setup mock Anthropic client
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
        mock_final_response.content = [Mock(text="Python is a high-level programming language.")]
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Setup config with temp database
        mock_config.CHROMA_PATH = temp_chroma_db
        
        # Create RAG system
        rag_system = RAGSystem(mock_config)
        
        # Add some test data
        from document_processor import DocumentProcessor
        processor = DocumentProcessor(mock_config.CHUNK_SIZE, mock_config.CHUNK_OVERLAP)
        
        # Create a simple course
        course = Course(
            title="Python Basics Course",
            instructor="Test Instructor"
        )
        course.lessons = [
            Lesson(
                lesson_number=0,
                title="Introduction"
            )
        ]
        
        chunks = [
            CourseChunk(
                course_title="Python Basics Course",
                lesson_number=0,
                content="Python is a high-level programming language known for its simplicity.",
                chunk_index=0
            )
        ]
        
        # Add to vector store
        rag_system.vector_store.add_course_metadata(course)
        rag_system.vector_store.add_course_content(chunks)
        
        # Perform query
        response, sources = rag_system.query("Tell me about Python basics")
        
        assert "Python is a high-level programming language" in response
        
        # Verify the flow
        assert mock_client.messages.create.call_count == 2
        
        # Check that tool was called
        first_call = mock_client.messages.create.call_args_list[0][1]
        assert "tools" in first_call
        assert first_call["tool_choice"] == {"type": "auto"}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])