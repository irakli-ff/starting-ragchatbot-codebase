"""
Test app factory to handle static file mounting issues in tests
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_app_for_testing(mock_rag_system=None):
    """
    Create a FastAPI app specifically for testing that avoids static file issues.
    This factory pattern allows tests to create isolated app instances.
    """
    from fastapi import HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional, Dict, Any, Union
    
    app = FastAPI(title="Test RAG System")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Use provided mock or create a new one
    rag_system = mock_rag_system or Mock()
    
    # Define request/response models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None
    
    class QueryResponse(BaseModel):
        answer: str
        sources: List[Union[str, Dict[str, str]]]
        session_id: str
    
    # Define endpoints
    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System"}
    
    @app.get("/api/health")
    async def health_check():
        return {"status": "healthy", "service": "rag-system"}
    
    @app.post("/api/query", response_model=QueryResponse)
    async def query_course(request: QueryRequest):
        try:
            # Use the injected RAG system
            response = rag_system.process_query(
                request.query,
                request.session_id
            )
            return QueryResponse(**response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/courses")
    async def get_course_list():
        try:
            courses = rag_system.get_all_courses()
            return {"courses": courses}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/stats")
    async def get_stats():
        try:
            stats = rag_system.get_stats()
            return stats
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app, rag_system


@pytest.fixture
def app_with_mocked_rag(mock_rag_system):
    """Fixture that provides app and RAG system for testing"""
    app, rag = create_app_for_testing(mock_rag_system)
    return app, rag


@pytest.fixture
def test_client_with_app(app_with_mocked_rag):
    """Fixture that provides a test client with the app"""
    app, rag = app_with_mocked_rag
    client = TestClient(app)
    return client, rag


class TestAppFactory:
    """Test the app factory pattern"""
    
    def test_app_creation(self):
        """Test that app can be created without static file issues"""
        app, rag = create_app_for_testing()
        assert app is not None
        assert rag is not None
        
        # Test that app has expected routes
        routes = [route.path for route in app.routes]
        assert "/" in routes
        assert "/api/health" in routes
        assert "/api/query" in routes
        assert "/api/courses" in routes
        assert "/api/stats" in routes
    
    def test_app_with_custom_rag(self, mock_rag_system):
        """Test app creation with custom RAG system"""
        app, rag = create_app_for_testing(mock_rag_system)
        
        # Verify the injected RAG system is used
        assert rag == mock_rag_system
        
        # Test with client
        client = TestClient(app)
        response = client.get("/api/health")
        assert response.status_code == 200
    
    def test_isolated_app_instances(self):
        """Test that multiple app instances are isolated"""
        app1, rag1 = create_app_for_testing()
        app2, rag2 = create_app_for_testing()
        
        # Apps should be different instances
        assert app1 != app2
        assert rag1 != rag2
        
        # Each should have its own RAG system mock
        rag1.test_value = "app1"
        rag2.test_value = "app2"
        
        assert rag1.test_value != rag2.test_value


class TestIntegrationWithFactory:
    """Integration tests using the factory pattern"""
    
    @pytest.mark.api
    def test_full_query_flow(self, test_client_with_app):
        """Test complete query flow with factory-created app"""
        client, mock_rag = test_client_with_app
        
        # Configure mock response
        mock_rag.process_query.return_value = {
            "answer": "Test answer from factory app",
            "sources": [{"text": "Source 1", "link": "https://example.com"}],
            "session_id": "factory-session-123"
        }
        
        # Make request
        response = client.post("/api/query", json={
            "query": "Test query",
            "session_id": None
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Test answer from factory app"
        assert data["session_id"] == "factory-session-123"
        
        # Verify mock was called
        mock_rag.process_query.assert_called_once_with("Test query", None)
    
    @pytest.mark.api
    def test_error_handling_in_factory_app(self, test_client_with_app):
        """Test error handling in factory-created app"""
        client, mock_rag = test_client_with_app
        
        # Configure mock to raise exception
        mock_rag.process_query.side_effect = Exception("Test error from factory")
        
        response = client.post("/api/query", json={
            "query": "Error query",
            "session_id": None
        })
        
        assert response.status_code == 500
        assert "Test error from factory" in response.json()["detail"]
    
    @pytest.mark.api
    def test_courses_endpoint_with_factory(self, test_client_with_app):
        """Test courses endpoint with factory app"""
        client, mock_rag = test_client_with_app
        
        mock_rag.get_all_courses.return_value = [
            {"title": "Course 1", "instructor": "Teacher 1"},
            {"title": "Course 2", "instructor": "Teacher 2"}
        ]
        
        response = client.get("/api/courses")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["courses"]) == 2
        assert data["courses"][0]["title"] == "Course 1"


def test_app_import_isolation():
    """
    Test that the factory pattern avoids import issues.
    This test verifies that we can create an app without importing
    the main app.py file which has static file mounting issues.
    """
    # This should work without any import errors
    app, rag = create_app_for_testing()
    client = TestClient(app)
    
    # Basic smoke test
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Course Materials RAG System"


if __name__ == "__main__":
    # Run a quick test to verify the factory works
    print("Testing app factory...")
    app, rag = create_app_for_testing()
    print(f"✅ App created successfully: {app.title}")
    
    client = TestClient(app)
    response = client.get("/api/health")
    print(f"✅ Health check: {response.json()}")
    
    print("\nFactory pattern working correctly!")