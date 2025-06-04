#!/usr/bin/env python3
"""
AI Vision Analyzer for Survey Images

Enhanced version with OpenAI Vision API integration for actual AI-powered image analysis.
Provides detailed image descriptions, object detection, and sentiment analysis.
"""

import os
import base64
import json
from typing import Dict, List, Optional
import openai
from PIL import Image
import requests
from google_drive_image_analyzer import GoogleDriveImageAnalyzer

class AIVisionAnalyzer(GoogleDriveImageAnalyzer):
    """Enhanced analyzer with AI vision capabilities"""
    
    def __init__(self, csv_file_path: str, output_dir: str = "survey_analysis", openai_api_key: Optional[str] = None):
        super().__init__(csv_file_path, output_dir)
        
        # Set up OpenAI client
        if openai_api_key:
            openai.api_key = openai_api_key
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai.api_key:
            print("⚠️ Warning: No OpenAI API key found. AI analysis will be limited.")
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string for OpenAI API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def analyze_image_with_openai(self, image_path: str, survey_context: Dict) -> Dict:
        """Analyze image using OpenAI Vision API"""
        try:
            # Initialize analysis structure
            analysis = {
                "timestamp": self._get_timestamp(),
                "image_path": image_path,
                "ai_analysis": {
                    "description": "Image analysis placeholder - integrate with OpenAI Vision API",
                    "sentiment": "neutral",
                    "categories": [],
                    "keywords_detected": {}
                }
            }
            
            # Get base64 encoded image
            base64_image = self.encode_image_to_base64(image_path)
            
            # Prepare messages for OpenAI
            messages = [
                {
                    "role": "system",
                    "content": "You are an AI assistant that analyzes images from survey responses. Provide detailed, objective descriptions focusing on visual elements, context, and potential insights relevant to the survey."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Please analyze this image in the context of the following survey response:\n\n{self._format_survey_context(survey_context)}\n\nProvide a detailed description of what you see, focusing on elements that might be relevant to understanding the survey response."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=messages,
                max_tokens=500
            )
            
            # Extract AI description
            ai_description = response.choices[0].message.content
            
            # Update analysis with AI description
            analysis["ai_analysis"]["description"] = ai_description
            
            # Extract structured insights
            insights = self._extract_insights_from_description(ai_description, survey_context)
            analysis["ai_analysis"].update(insights)
            
            print(f"✅ AI analysis completed for {image_path}")
            return analysis
            
        except Exception as e:
            print(f"❌ Error in AI analysis for {image_path}: {e}")
            # Return a fallback result instead of calling analyze_image_with_ai to avoid recursion
            return {
                "timestamp": self._get_timestamp(),
                "image_path": image_path,
                "ai_analysis": {
                    "description": f"AI analysis failed: {e}",
                    "sentiment": "unknown",
                    "categories": [],
                    "keywords_detected": {}
                }
            }
    
    def _format_survey_context(self, survey_context: Dict) -> str:
        """Format survey context for AI prompt"""
        formatted = ""
        for key, value in survey_context.items():
            if key != "What apps are you using " and not key.startswith("http"):  # Skip URLs
                formatted += f"{key}: {value}\n"
        return formatted
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _extract_insights_from_description(self, description: str, survey_context: Dict) -> Dict:
        """
        Extract structured insights from AI description
        
        Args:
            description: AI-generated description
            survey_context: Survey response context
            
        Returns:
            Structured insights dictionary
        """
        # Basic keyword extraction and sentiment analysis
        positive_keywords = ['good', 'great', 'excellent', 'beautiful', 'amazing', 'wonderful', 'happy', 'positive']
        negative_keywords = ['bad', 'poor', 'terrible', 'ugly', 'awful', 'sad', 'negative', 'disappointing']
        
        description_lower = description.lower()
        
        # Sentiment analysis
        positive_count = sum(1 for word in positive_keywords if word in description_lower)
        negative_count = sum(1 for word in negative_keywords if word in description_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Extract potential categories
        tech_keywords = ['phone', 'computer', 'app', 'software', 'digital', 'screen', 'interface', 'ui', 'ux']
        lifestyle_keywords = ['home', 'food', 'travel', 'fashion', 'fitness', 'hobby', 'leisure']
        social_keywords = ['people', 'friends', 'family', 'social', 'community', 'group']
        
        categories = []
        if any(word in description_lower for word in tech_keywords):
            categories.append("technology")
        if any(word in description_lower for word in lifestyle_keywords):
            categories.append("lifestyle")
        if any(word in description_lower for word in social_keywords):
            categories.append("social")
        
        return {
            "sentiment": sentiment,
            "sentiment_confidence": abs(positive_count - negative_count) / max(positive_count + negative_count, 1),
            "categories": categories,
            "keywords_detected": {
                "positive": positive_count,
                "negative": negative_count,
                "tech": sum(1 for word in tech_keywords if word in description_lower),
                "lifestyle": sum(1 for word in lifestyle_keywords if word in description_lower),
                "social": sum(1 for word in social_keywords if word in description_lower)
            }
        }
    
    def analyze_image_with_ai(self, image_path: str, survey_context: Dict) -> Dict:
        """Override parent method to use OpenAI when available"""
        if openai.api_key:
            return self.analyze_image_with_openai(image_path, survey_context)
        else:
            return super().analyze_image_with_ai(image_path, survey_context)
    
    def generate_ai_insights_summary(self, analysis_data: Dict) -> Dict:
        """
        Generate comprehensive AI insights summary from all analyses
        
        Args:
            analysis_data: Complete analysis data
            
        Returns:
            AI insights summary
        """
        individual_analyses = analysis_data.get('individual_analyses', {})
        
        if not individual_analyses:
            return {"error": "No individual analyses found"}
        
        # Aggregate insights
        all_sentiments = []
        all_categories = []
        tech_mentions = 0
        lifestyle_mentions = 0
        social_mentions = 0
        
        detailed_descriptions = []
        
        for idx, analysis in individual_analyses.items():
            ai_analysis = analysis.get('ai_analysis', {})
            
            # Collect sentiments
            sentiment = ai_analysis.get('sentiment', 'neutral')
            all_sentiments.append(sentiment)
            
            # Collect categories
            categories = ai_analysis.get('categories', [])
            all_categories.extend(categories)
            
            # Collect keyword counts
            keywords = ai_analysis.get('keywords_detected', {})
            tech_mentions += keywords.get('tech', 0)
            lifestyle_mentions += keywords.get('lifestyle', 0)
            social_mentions += keywords.get('social', 0)
            
            # Collect descriptions
            description = ai_analysis.get('description', '')
            if description and description != "Image analysis placeholder - integrate with OpenAI Vision API":
                detailed_descriptions.append({
                    'response_id': idx,
                    'description': description
                })
        
        # Calculate statistics
        total_responses = len(individual_analyses)
        positive_sentiment_pct = (all_sentiments.count('positive') / total_responses) * 100
        negative_sentiment_pct = (all_sentiments.count('negative') / total_responses) * 100
        neutral_sentiment_pct = (all_sentiments.count('neutral') / total_responses) * 100
        
        # Top categories
        from collections import Counter
        category_counts = Counter(all_categories)
        top_categories = category_counts.most_common(5)
        
        insights_summary = {
            "total_images_analyzed": total_responses,
            "sentiment_distribution": {
                "positive": f"{positive_sentiment_pct:.1f}%",
                "negative": f"{negative_sentiment_pct:.1f}%",
                "neutral": f"{neutral_sentiment_pct:.1f}%"
            },
            "top_categories": top_categories,
            "content_themes": {
                "technology_mentions": tech_mentions,
                "lifestyle_mentions": lifestyle_mentions,
                "social_mentions": social_mentions
            },
            "key_insights": self._generate_key_insights(all_sentiments, all_categories, tech_mentions, lifestyle_mentions, social_mentions),
            "detailed_descriptions": detailed_descriptions
        }
        
        return insights_summary
    
    def _generate_key_insights(self, sentiments: List[str], categories: List[str], tech: int, lifestyle: int, social: int) -> List[str]:
        """Generate key insights from aggregated data"""
        insights = []
        
        # Sentiment insights
        positive_pct = (sentiments.count('positive') / len(sentiments)) * 100
        if positive_pct > 60:
            insights.append(f"High positive sentiment ({positive_pct:.1f}%) suggests users are generally satisfied")
        elif positive_pct < 30:
            insights.append(f"Low positive sentiment ({positive_pct:.1f}%) may indicate areas for improvement")
        
        # Category insights
        from collections import Counter
        category_counts = Counter(categories)
        if category_counts:
            top_category = category_counts.most_common(1)[0]
            insights.append(f"'{top_category[0]}' is the dominant theme ({top_category[1]} mentions)")
        
        # Content insights
        total_mentions = tech + lifestyle + social
        if total_mentions > 0:
            if tech / total_mentions > 0.4:
                insights.append("Strong technology focus detected in submitted images")
            if lifestyle / total_mentions > 0.4:
                insights.append("Lifestyle content is prominent in user submissions")
            if social / total_mentions > 0.4:
                insights.append("Social elements are frequently featured")
        
        return insights

def main():
    """Main function for AI Vision Analyzer"""
    print("🤖 AI Vision Analyzer for Survey Data")
    print("=" * 50)
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenAI API key (or press Enter to skip AI analysis): ").strip()
    
    # Initialize analyzer
    csv_file = "googleFormExample.csv"
    analyzer = AIVisionAnalyzer(csv_file, openai_api_key=api_key)
    
    # Load and process data
    data = analyzer.load_survey_data()
    if data is None:
        return
    
    print(f"\n📋 Survey Data Preview:")
    print(data.head())
    
    # Download images
    print(f"\n🔄 Downloading images from Google Drive...")
    downloaded_images = analyzer.download_all_images()
    
    if downloaded_images:
        print(f"\n🔍 Running AI analysis on downloaded images...")
        analysis_data = analyzer.generate_comprehensive_analysis()
        
        # Generate AI insights summary
        if api_key:
            print(f"\n🧠 Generating AI insights summary...")
            insights = analyzer.generate_ai_insights_summary(analysis_data)
            
            # Save insights
            insights_file = analyzer.analysis_dir / f"ai_insights_{analyzer._get_timestamp().replace(':', '-')}.json"
            with open(insights_file, 'w') as f:
                json.dump(insights, f, indent=2)
            
            print(f"📊 AI insights saved to: {insights_file}")
            
            # Display key insights
            print(f"\n🔍 Key Insights:")
            for insight in insights.get('key_insights', []):
                print(f"  • {insight}")
        
        report_path = analyzer.create_analysis_report()
        print(f"\n🎉 Analysis complete! Report saved to: {report_path}")
    else:
        print("\n❌ No images were downloaded successfully")

if __name__ == "__main__":
    main() 