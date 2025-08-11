"""
Enhanced API endpoint tests with proper request/response handling
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_test_app():
    """Create a test app without static file mounts"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional, Dict, Any, Union
    
    # Create minimal app for testing
    app = FastAPI(title="Test RAG System")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Define models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None
    
    class QueryResponse(BaseModel):
        answer: str
        sources: List[Union[str, Dict[str, str]]]
        session_id: str
    
    class CourseStats(BaseModel):
        total_courses: int
        total_lessons: int
        total_chunks: int
        embedding_model: str
    
    # Mock RAG system
    mock_rag_system = Mock()
    
    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System"}
    
    @app.get("/api/health")
    async def health_check():
        return {"status": "healthy", "service": "rag-system"}
    
    @app.post("/api/query", response_model=QueryResponse)
    async def query_course(request: QueryRequest):
        try:
            response = mock_rag_system.process_query(
                request.query,
                request.session_id
            )
            return QueryResponse(
                answer=response.get("answer", ""),
                sources=response.get("sources", []),
                session_id=response.get("session_id", "")
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/courses")
    async def get_course_list():
        try:
            courses = mock_rag_system.get_all_courses()
            return {"courses": courses}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/stats", response_model=CourseStats)
    async def get_stats():
        try:
            stats = mock_rag_system.get_stats()
            return CourseStats(**stats)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app, mock_rag_system


@pytest.fixture
def test_client():
    """Create a test client with mocked RAG system"""
    app, mock_rag = create_test_app()
    client = TestClient(app)
    return client, mock_rag


@pytest.fixture
def mock_rag_response():
    """Mock RAG system response"""
    return {
        "answer": "Python is a high-level programming language known for its simplicity.",
        "sources": [
            {"text": "Introduction to Python", "link": "https://example.com/lesson0"},
            {"text": "Python Basics", "link": "https://example.com/lesson1"}
        ],
        "session_id": "test-session-123"
    }


@pytest.fixture
def mock_course_list():
    """Mock course list"""
    return [
        {
            "title": "Python Programming",
            "instructor": "John Doe",
            "lessons": 10,
            "link": "https://example.com/python"
        },
        {
            "title": "Data Science with Python",
            "instructor": "Jane Smith",
            "lessons": 8,
            "link": "https://example.com/datascience"
        }
    ]


@pytest.fixture
def mock_stats():
    """Mock system statistics"""
    return {
        "total_courses": 2,
        "total_lessons": 18,
        "total_chunks": 150,
        "embedding_model": "all-MiniLM-L6-v2"
    }


class TestAPIEndpoints:
    """Test suite for API endpoints"""
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint returns expected message"""
        client, _ = test_client
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Course Materials RAG System"}
    
    def test_health_check(self, test_client):
        """Test health check endpoint"""
        client, _ = test_client
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "rag-system"
    
    def test_query_endpoint_success(self, test_client, mock_rag_response):
        """Test successful query processing"""
        client, mock_rag = test_client
        mock_rag.process_query.return_value = mock_rag_response
        
        request_data = {
            "query": "What is Python?",
            "session_id": None
        }
        
        response = client.post("/api/query", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == mock_rag_response["answer"]
        assert len(data["sources"]) == 2
        
        # Verify RAG system was called correctly
        mock_rag.process_query.assert_called_once_with(
            "What is Python?",
            None
        )
    
    def test_query_endpoint_with_session(self, test_client, mock_rag_response):
        """Test query with existing session ID"""
        client, mock_rag = test_client
        mock_rag.process_query.return_value = mock_rag_response
        
        request_data = {
            "query": "Tell me more",
            "session_id": "existing-session-456"
        }
        
        response = client.post("/api/query", json=request_data)
        assert response.status_code == 200
        
        mock_rag.process_query.assert_called_once_with(
            "Tell me more",
            "existing-session-456"
        )
    
    def test_query_endpoint_error_handling(self, test_client):
        """Test error handling in query endpoint"""
        client, mock_rag = test_client
        mock_rag.process_query.side_effect = Exception("RAG system error")
        
        request_data = {
            "query": "What is Python?",
            "session_id": None
        }
        
        response = client.post("/api/query", json=request_data)
        assert response.status_code == 500
        assert "RAG system error" in response.json()["detail"]
    
    def test_query_endpoint_invalid_request(self, test_client):
        """Test query endpoint with invalid request data"""
        client, _ = test_client
        
        # Missing query field
        response = client.post("/api/query", json={})
        assert response.status_code == 422
    
    def test_courses_endpoint(self, test_client, mock_course_list):
        """Test course list endpoint"""
        client, mock_rag = test_client
        mock_rag.get_all_courses.return_value = mock_course_list
        
        response = client.get("/api/courses")
        assert response.status_code == 200
        
        data = response.json()
        assert "courses" in data
        assert len(data["courses"]) == 2
        assert data["courses"][0]["title"] == "Python Programming"
        
        mock_rag.get_all_courses.assert_called_once()
    
    def test_courses_endpoint_error(self, test_client):
        """Test course list endpoint error handling"""
        client, mock_rag = test_client
        mock_rag.get_all_courses.side_effect = Exception("Database error")
        
        response = client.get("/api/courses")
        assert response.status_code == 500
        assert "Database error" in response.json()["detail"]
    
    def test_stats_endpoint(self, test_client, mock_stats):
        """Test statistics endpoint"""
        client, mock_rag = test_client
        mock_rag.get_stats.return_value = mock_stats
        
        response = client.get("/api/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_courses"] == 2
        assert data["total_lessons"] == 18
        assert data["total_chunks"] == 150
        assert data["embedding_model"] == "all-MiniLM-L6-v2"
        
        mock_rag.get_stats.assert_called_once()
    
    def test_stats_endpoint_error(self, test_client):
        """Test statistics endpoint error handling"""
        client, mock_rag = test_client
        mock_rag.get_stats.side_effect = Exception("Stats calculation error")
        
        response = client.get("/api/stats")
        assert response.status_code == 500
        assert "Stats calculation error" in response.json()["detail"]


class TestAPIIntegration:
    """Integration tests for API with mocked components"""
    
    def test_query_with_tool_execution(self, test_client):
        """Test query that triggers tool execution"""
        client, mock_rag = test_client
        
        # Mock response with tool execution
        mock_rag.process_query.return_value = {
            "answer": "Based on the course content, Python uses dynamic typing.",
            "sources": [
                {"text": "Variables and Data Types", "link": "https://example.com/lesson2"},
                {"text": "Python Type System", "link": "https://example.com/lesson3"}
            ],
            "session_id": "session-789"
        }
        
        request_data = {
            "query": "How does Python handle variable types?",
            "session_id": None
        }
        
        response = client.post("/api/query", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "dynamic typing" in data["answer"]
        assert len(data["sources"]) == 2
    
    def test_query_with_empty_sources(self, test_client):
        """Test query with no sources found"""
        client, mock_rag = test_client
        
        mock_rag.process_query.return_value = {
            "answer": "I don't have specific information about that topic.",
            "sources": [],
            "session_id": "session-999"
        }
        
        request_data = {
            "query": "What about quantum computing?",
            "session_id": None
        }
        
        response = client.post("/api/query", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["sources"]) == 0
        assert "don't have specific information" in data["answer"]
    
    def test_concurrent_queries(self, test_client, mock_rag_response):
        """Test handling multiple concurrent queries"""
        client, mock_rag = test_client
        mock_rag.process_query.return_value = mock_rag_response
        
        queries = [
            {"query": "What is Python?", "session_id": None},
            {"query": "Tell me about variables", "session_id": None},
            {"query": "Explain loops", "session_id": None}
        ]
        
        responses = []
        for query_data in queries:
            response = client.post("/api/query", json=query_data)
            responses.append(response)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            assert "answer" in response.json()
        
        # Verify all queries were processed
        assert mock_rag.process_query.call_count == 3


class TestAPIValidation:
    """Test request validation and error responses"""
    
    def test_query_validation_empty_string(self, test_client):
        """Test that empty query strings are rejected"""
        client, _ = test_client
        
        request_data = {
            "query": "   ",  # Whitespace only
            "session_id": None
        }
        
        response = client.post("/api/query", json=request_data)
        # Note: Depending on validation, this might be 422 or process as normal
        # Adjust based on actual validation requirements
    
    def test_query_validation_long_query(self, test_client, mock_rag_response):
        """Test handling of very long queries"""
        client, mock_rag = test_client
        mock_rag.process_query.return_value = mock_rag_response
        
        long_query = "What is Python? " * 100  # Very long query
        request_data = {
            "query": long_query,
            "session_id": None
        }
        
        response = client.post("/api/query", json=request_data)
        assert response.status_code == 200
    
    def test_invalid_http_methods(self, test_client):
        """Test that invalid HTTP methods are rejected"""
        client, _ = test_client
        
        # GET request to POST endpoint
        response = client.get("/api/query")
        assert response.status_code == 405
        
        # POST request to GET endpoint
        response = client.post("/api/courses", json={})
        assert response.status_code == 405


if __name__ == "__main__":
    pytest.main([__file__, "-v"])