# This script is used to test the HateSpeechDetectionAgent class.

import asyncio
from app.agents.hate_speech_detection import HateSpeechDetectionAgent


async def main():
    agent = HateSpeechDetectionAgent()
    test_text = "I hate u."
    print(f"Testing hate speech detection on: {test_text!r}")

    # No need to await if classify_text is not async
    result = agent.classify_text(test_text)
    print("\nClassification Result:")
    print(f"Label: {result['classification']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Reason: {result['reason']}")


if __name__ == "__main__":
    asyncio.run(main())
