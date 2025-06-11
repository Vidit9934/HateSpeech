# error_handler.py

import asyncio
import logging
from app.agents.error_handler import ErrorHandlerAgent, ErrorType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ErrorHandlerTest")

async def test_error_handler():
    print("\n🔍 Testing ErrorHandlerAgent")
    print("=" * 50)
    
    # Initialize agent
    handler = ErrorHandlerAgent()
    
    # Test cases with different error types and contexts
    test_cases = [
        {
            "name": "Classification Error",
            "error": ValueError("Invalid text format"),
            "type": ErrorType.CLASSIFICATION_ERROR,
            "context": {"text": "test input", "model": "gpt-4"}
        },
        {
            "name": "API Error",
            "error": ConnectionError("Failed to connect to API"),
            "type": ErrorType.API_ERROR,
            "context": {"endpoint": "azure-openai", "attempt": 3}
        },
        {
            "name": "Retrieval Error",
            "error": Exception("Failed to fetch policies"),
            "type": ErrorType.RETRIEVAL_ERROR,
            "context": {"query": "hate speech", "collection": "policies"}
        },
        {
            "name": "System Error",
            "error": SystemError("Critical system failure"),
            "type": ErrorType.SYSTEM_ERROR,
            "context": {"component": "database", "operation": "read"}
        }
    ]
    
    # Run tests
    for case in test_cases:
        print(f"\n📋 Testing: {case['name']}")
        print("-" * 30)
        
        try:
            result = handler.handle_error(
                error=case["error"],
                error_type=case["type"],
                context=case["context"]
            )
            
            # Print result details
            print(f"Error Type: {result['error_info']['type']}")
            print(f"Timestamp: {result['error_info']['timestamp']}")
            print(f"Message: {result['error_info']['message']}")
            
            if "action" in result:
                print(f"Action: {result['action']}")
            if "message" in result:
                print(f"Response Message: {result['message']}")
                
            print("✅ Error handled successfully")
            
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")

async def test_emergency_fallback():
    print("\n🔍 Testing Emergency Fallback")
    print("=" * 50)
    
    handler = ErrorHandlerAgent()
    
    # Force an error in the error handler itself
    try:
        # Create an invalid context that will cause the handler to fail
        invalid_context = {"invalid": lambda x: x}  # Cannot be serialized
        
        result = handler.handle_error(
            error=Exception("Test error"),
            error_type=ErrorType.SYSTEM_ERROR,
            context=invalid_context
        )
        
        print("Response:", result)
        print("✅ Emergency fallback worked")
        
    except Exception as e:
        print(f"❌ Emergency fallback failed: {str(e)}")

async def main():
    print("🚀 Starting Error Handler Tests")
    print("=" * 50)
    
    await test_error_handler()
    await test_emergency_fallback()
    
    print("\n✨ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())