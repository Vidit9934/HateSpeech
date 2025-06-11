from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from typing import Dict, List, Any
from app.utils.config import Config
from app.models.schemas import PolicyReasoning
import logging

logger = logging.getLogger("PolicyReasoningAgent")


class PolicyReasoningAgent:
    def __init__(self):
        try:
            self.model = AzureChatOpenAI(
                openai_api_version=Config.DIAL_API_VERSION,
                azure_deployment=Config.PRIMARY_MODEL_NAME,
                azure_endpoint=Config.DIAL_API_ENDPOINT,
                api_key=Config.DIAL_API_KEY,
                temperature=0.2,
            )
            logger.info("Successfully initialized PolicyReasoningAgent with Azure OpenAI")
        except Exception as e:
            logger.error(f"Failed to initialize PolicyReasoningAgent: {str(e)}")
            raise

    async def generate_policy_reasoning(
        self,
        original_text: str,
        classification_result: Dict,
        relevant_policies: List[Dict]
    ) -> Dict[str, Any]:
        """Generate reasoning about policy violations"""
        try:
            # Format policies for prompt
            policies_text = "\n".join([
                f"Policy {i+1}: {p['content']}" 
                for i, p in enumerate(relevant_policies)
            ])

            # Create prompt
            prompt = f"""
            Analyze this content against our policies:
            
            Content: "{original_text}"
            
            Classification: {classification_result['classification']} 
            (Confidence: {classification_result['confidence']})
            
            Relevant Policies:
            {policies_text}
            
            Provide a structured analysis with these sections:
            - Policy Violations (list specific violations)
            - Detailed Reasoning (explain how content violates policies)
            - Supporting Evidence (quote relevant policy sections)
            - Severity Assessment (assess impact and severity)
            - Additional Context (any other relevant factors)
            """

            # Get AI response
            messages = [
                SystemMessage(content="You are a policy analysis expert."),
                HumanMessage(content=prompt)
            ]
            response = await self.model.ainvoke(messages)
            logger.info(f"Raw OpenAI Response:\n{response.content}")

            # Parse the response
            parsed = self._parse_response(response.content)
            
            # Combine with original classification
            result = {
                "policy_violations": parsed["policy_violations"],
                "detailed_reasoning": parsed["detailed_reasoning"],
                "supporting_evidence": parsed["supporting_evidence"],
                "severity_assessment": parsed["severity_assessment"],
                "additional_context": parsed["additional_context"],
                "original_classification": classification_result,
                "policies_analyzed": len(relevant_policies)
            }

            logger.info("Successfully generated policy reasoning")
            return result

        except Exception as e:
            logger.error(f"Error in generate_reasoning: {str(e)}")
            return self._get_fallback_reasoning(classification_result)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the Markdown-formatted OpenAI response into structured sections"""
        sections = {
            "policy_violations": [],
            "detailed_reasoning": "",
            "supporting_evidence": "",
            "severity_assessment": "",
            "additional_context": ""
        }
        
        # Map markdown headers to our section keys
        header_map = {
            "### Policy Violations": "policy_violations",
            "### Detailed Reasoning": "detailed_reasoning",
            "### Supporting Evidence": "supporting_evidence",
            "### Severity Assessment": "severity_assessment",
            "### Additional Context": "additional_context"
        }
        
        current_section = None
        content_buffer = []
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if this line is a header
            for header, section in header_map.items():
                if line.startswith(header):
                    # Save previous section content
                    if current_section and content_buffer:
                        if current_section == "policy_violations":
                            # Extract numbered/bulleted items
                            for item in content_buffer:
                                # Remove markdown list indicators and clean
                                clean_item = item.lstrip('1234567890.- *').strip()
                                if clean_item:
                                    sections[current_section].append(clean_item)
                        else:
                            sections[current_section] = ' '.join(content_buffer)
                    # Start new section
                    current_section = section
                    content_buffer = []
                    break
            else:
                # Not a header, so add to current section if we have one
                if current_section:
                    # Clean markdown formatting
                    clean_line = line.strip('*').strip()
                    if clean_line:
                        content_buffer.append(clean_line)
        
        # Don't forget to save the last section
        if current_section and content_buffer:
            if current_section == "policy_violations":
                for item in content_buffer:
                    clean_item = item.lstrip('1234567890.- *').strip()
                    if clean_item:
                        sections[current_section].append(clean_item)
            else:
                sections[current_section] = ' '.join(content_buffer)
        
        # Ensure each section has content
        for key, value in sections.items():
            if isinstance(value, list) and not value:
                sections[key] = ["No violations detected"]
            elif isinstance(value, str) and not value:
                sections[key] = "No information available"
                
        return sections

    def _get_fallback_reasoning(self, classification_result: Dict) -> Dict[str, Any]:
        """Provide fallback reasoning when parsing fails"""
        return {
            "policy_violations": ["Unable to determine specific violations"],
            "detailed_reasoning": "Analysis failed - using fallback reasoning",
            "supporting_evidence": "No evidence could be extracted",
            "severity_assessment": "Unable to assess severity",
            "additional_context": "Error occurred during analysis",
            "original_classification": classification_result,
            "policies_analyzed": 0
        }
