"""
OpenAI API utilities for the Comprehensive Survey Analyzer
"""

import json
from typing import Dict, List, Optional
import openai
from config import OPENAI_API_KEY

class OpenAIClient:
    """Handles all OpenAI API interactions"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenAI client"""
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Set up OpenAI client
        try:
            version = getattr(openai, "__version__", "0")
            major = int(version.split(".")[0]) if version and version[0].isdigit() else 0
            if major >= 1:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                self.use_new_api = True
            else:
                openai.api_key = self.api_key
                self.use_new_api = False
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
    
    def chat_completion(self, messages: List[Dict], model: str = "gpt-3.5-turbo", temperature: float = 0.3) -> str:
        """Call OpenAI chat completion with backward-compatible logic"""
        try:
            if self.use_new_api:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                return resp.choices[0].message.content
            else:
                resp = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                return resp["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"OpenAI chat completion failed: {e}")
    
    def generate_text_insights(self, question: str, responses: List[str]) -> Dict:
        """Generate insights from text responses"""
        try:
            # Sample up to first 100 responses to reduce token usage
            sampled = responses[:100]
            sample_text = "\n".join(f"- {r.strip()}" for r in sampled)
            
            prompt = (
                f"You are an expert survey analyst. Based on the following responses to the question '{question}', "
                "write a concise summary (3-5 sentences) capturing the main themes, then provide a bullet list of 3-5 "
                "actionable recommendations an organization could take. Return the result as JSON with keys 'summary' "
                "and 'actionable_insights' (array of strings).\n\nResponses:\n" + sample_text
            )
            
            content = self.chat_completion([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ])
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"summary": content.strip(), "actionable_insights": []}
                
        except Exception as e:
            return {"summary": f"Error generating insights: {e}", "actionable_insights": []}
    
    def generate_vc_trend_insights(self, survey_data: Dict, text_columns: List[str], prompt: str) -> str:
        """Generate VC trend insights from survey data"""
        try:
            # Build context: list up to 200 responses per column to control tokens
            parts = []
            for col in text_columns:
                responses = survey_data[col][:200] if col in survey_data else []
                if responses:
                    joined = "\n".join(f"- {r.strip()}" for r in responses)
                    parts.append(f"\n\n### Question: {col}\n{joined}")
            
            context_text = "".join(parts)
            full_prompt = prompt + "\n\nDATA:" + context_text + "\n\nReturn your analysis as Markdown."
            
            return self.chat_completion([
                {"role": "user", "content": full_prompt}
            ], model="gpt-3.5-turbo-16k", temperature=0.2)
            
        except Exception as e:
            return f"Error generating VC insights: {e}"

# Create a global OpenAI client instance
openai_client = OpenAIClient() 