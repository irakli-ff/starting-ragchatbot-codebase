"""
Test with actual API call to identify the real issue
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from rag_system import RAGSystem


def test_real_api_call():
    """Test with actual Anthropic API to identify failure"""
    config = Config()
    
    print(f"\n[LIVE TEST] Testing with real API")
    print(f"[LIVE TEST] API Key present: {bool(config.ANTHROPIC_API_KEY)}")
    
    if not config.ANTHROPIC_API_KEY:
        print("[LIVE TEST] ERROR: No API key found!")
        print("[LIVE TEST] Please set ANTHROPIC_API_KEY in .env file")
        return False
    
    try:
        # Create RAG system
        rag_system = RAGSystem(config)
        
        # Load documents
        print("[LIVE TEST] Loading documents...")
        courses, chunks = rag_system.add_course_folder("../docs", clear_existing=False)
        print(f"[LIVE TEST] Loaded {courses} new courses with {chunks} chunks")
        
        # Test a simple query
        print("[LIVE TEST] Testing query: 'What is Python?'")
        response, sources = rag_system.query("What is Python?")
        
        print(f"[LIVE TEST] Response: {response[:200]}...")
        print(f"[LIVE TEST] Sources: {sources}")
        
        return True
        
    except Exception as e:
        print(f"[LIVE TEST] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_real_api_call()
    if success:
        print("\n✅ API call successful!")
    else:
        print("\n❌ API call failed - check the error above")