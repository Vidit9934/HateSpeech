import asyncio
import logging
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
from ..models.schemas import ClassificationResult, ClassificationLabel
from ..utils.config import settings

logger = logging.getLogger(__name__)

class HateSpeechDetectionAgent:
    """
    Agent responsible for detecting hate speech using OpenAI API.
    Classifies text into 5 categories: Hate, Toxic, Offensive, Neutral, Ambiguous
    """
    
    def __init__(self, api_key: str = None):
        self.client = AsyncOpenAI(api_key=api_key or settings.OPENAI_API_KEY)
        self.model = "gpt-4"
        
    async def classify_text(self, text: str) -> ClassificationResult:
        """
        Classify input text for hate speech detection
        
        Args:
            text: The text to classify
            
        Returns:
            ClassificationResult with label, confidence, and explanation
        """
        try:
            prompt = self._create_classification_prompt(text)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result = self._parse_response(response.choices[0].message.content)
            logger.info(f"Text classified as: {result.label} with confidence: {result.confidence}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in hate speech classification: {str(e)}")
            # Return fallback result
            return ClassificationResult(
                label=ClassificationLabel.AMBIGUOUS,
                confidence=0.0,
                explanation="Classification failed due to technical error",
                reasoning_steps=["Error occurred during classification"],
                policy_violations=[]
            )
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for classification"""
        return """You are an expert content moderation AI. Your task is to classify text into exactly one of these categories:

1. HATE: Content that attacks, dehumanizes, or promotes violence against individuals or groups based on protected characteristics (race, religion, ethnicity, gender, sexual orientation, disability, etc.)

2. TOXIC: Content that is harmful, disrespectful, or likely to make someone leave a conversation. Includes severe harassment, threats, or extremely offensive language.

3. OFFENSIVE: Content that is rude, disrespectful, or inappropriate but doesn't rise to the level of hate speech or toxicity. Includes mild profanity or insults.

4. NEUTRAL: Content that is normal, acceptable, and doesn't violate any guidelines. Safe for general audiences.

5. AMBIGUOUS: Content that is unclear, context-dependent, or borderline between categories. Requires human review.

You must respond in this exact JSON format:
{
    "label": "HATE|TOXIC|OFFENSIVE|NEUTRAL|AMBIGUOUS",
    "confidence": 0.0-1.0,
    "explanation": "Brief explanation of the classification",
    "reasoning_steps": ["step1", "step2", "step3"],
    "policy_violations": ["specific policy violations if any"]
}

Be consistent, accurate, and err on the side of caution for borderline cases."""

    def _create_classification_prompt(self, text: str) -> str:
        """Create the classification prompt"""
        return f"""Please classify the following text for content moderation:

Text to classify: "{text}"

Analyze this text carefully and provide your classification following the exact JSON format specified."""

    def _parse_response(self, response_text: str) -> ClassificationResult:
        """Parse the API response into ClassificationResult"""
        try:
            import json
            
            # Clean the response text
            cleaned_text = response_text.strip()
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            
            result_dict = json.loads(cleaned_text)
            
            # Map string label to enum
            label_mapping = {
                "HATE": ClassificationLabel.HATE,
                "TOXIC": ClassificationLabel.TOXIC,
                "OFFENSIVE": ClassificationLabel.OFFENSIVE,
                "NEUTRAL": ClassificationLabel.NEUTRAL,
                "AMBIGUOUS": ClassificationLabel.AMBIGUOUS
            }
            
            label = label_mapping.get(result_dict["label"], ClassificationLabel.AMBIGUOUS)
            
            return ClassificationResult(
                label=label,
                confidence=float(result_dict.get("confidence", 0.5)),
                explanation=result_dict.get("explanation", "No explanation provided"),
                reasoning_steps=result_dict.get("reasoning_steps", []),
                policy_violations=result_dict.get("policy_violations", [])
            )
            
        except Exception as e:
            logger.error(f"Error parsing classification response: {str(e)}")
            return ClassificationResult(
                label=ClassificationLabel.AMBIGUOUS,
                confidence=0.0,
                explanation="Failed to parse classification result",
                reasoning_steps=["Response parsing failed"],
                policy_violations=[]
            )
    
    async def batch_classify(self, texts: list[str]) -> list[ClassificationResult]:
        """Classify multiple texts concurrently"""
        tasks = [self.classify_text(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in the results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error classifying text {i}: {str(result)}")
                processed_results.append(ClassificationResult(
                    label=ClassificationLabel.AMBIGUOUS,
                    confidence=0.0,
                    explanation="Batch classification failed",
                    reasoning_steps=["Batch processing error"],
                    policy_violations=[]
                ))
            else:
                processed_results.append(result)
        
        return processed_results