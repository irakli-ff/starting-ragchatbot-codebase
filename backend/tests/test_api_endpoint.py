"""
Test the API endpoint directly to diagnose the issue
"""

import json

import requests


def test_api_endpoint():
    """Test the /api/query endpoint directly"""

    # Check health first
    print("\n[API TEST] Checking health endpoint...")
    try:
        response = requests.get("http://localhost:8000/api/health")
        if response.status_code == 200:
            health = response.json()
            print(f"[API TEST] Health: {health}")
        else:
            print(f"[API TEST] Health check failed: {response.status_code}")
    except Exception as e:
        print(f"[API TEST] Cannot connect to server: {e}")
        print(
            "[API TEST] Make sure the server is running with: cd backend && uv run uvicorn app:app --reload --port 8000 --host 0.0.0.0"
        )
        return False

    # Test query endpoint
    print("\n[API TEST] Testing query endpoint...")

    test_queries = [
        "What is Python?",
        "Tell me about variables in Python",
        "What courses are available?",
        "Show me the course outline for MCP",
    ]

    for query in test_queries:
        print(f"\n[API TEST] Query: '{query}'")

        payload = {"query": query, "session_id": None}

        try:
            response = requests.post(
                "http://localhost:8000/api/query",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                data = response.json()
                print(f"[API TEST] ✅ Success!")
                print(f"[API TEST] Answer preview: {data['answer'][:100]}...")
                print(f"[API TEST] Sources: {len(data.get('sources', []))} sources")
            else:
                print(f"[API TEST] ❌ Failed with status: {response.status_code}")
                print(f"[API TEST] Error: {response.text}")

        except Exception as e:
            print(f"[API TEST] ❌ Request failed: {e}")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("API ENDPOINT TEST")
    print("=" * 60)
    test_api_endpoint()
