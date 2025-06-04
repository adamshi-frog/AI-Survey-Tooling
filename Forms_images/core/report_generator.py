"""
Report generation functionality for the Comprehensive Survey Analyzer
"""

from datetime import datetime
from typing import Dict, Optional
import pandas as pd

class ReportGenerator:
    """Handles generation of analysis reports"""
    
    def __init__(self, survey_data: pd.DataFrame, analysis: Dict):
        """Initialize the report generator"""
        self.survey_data = survey_data
        self.analysis = analysis
    
    def generate_report(self, drive_analysis: Optional[Dict] = None, 
                       ai_analysis: Optional[Dict] = None,
                       downloaded_images: Optional[Dict] = None,
                       ai_text_insights: Optional[Dict] = None,
                       vc_trend_insights: Optional[str] = None) -> str:
        """Generate a comprehensive analysis report"""
        
        report_content = f"""
# 🤖 Comprehensive Survey Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Executive Summary

**Survey Overview:**
- Total Responses: {self.analysis['total_responses']}
- Questions/Columns: {self.analysis['total_columns']}
- Data Quality Score: {100 - self.analysis['data_quality']['missing_data_percentage']:.1f}%

"""
        
        # Add image analysis section if available
        if drive_analysis and downloaded_images:
            report_content += f"""
**Image Analysis:**
- Google Drive Links Found: {drive_analysis.get('total_links', 0)}
- Images Successfully Downloaded: {len(downloaded_images)}
- Download Success Rate: {drive_analysis.get('accessibility_rate', '0%')}

"""
        
        # Add detailed analysis section
        report_content += """
## 🔍 Detailed Analysis

### Survey Data Structure
"""
        
        # Add column analysis
        report_content += "\n**Column Types:**\n"
        report_content += f"- Text Columns: {len(self.analysis['text_columns'])}\n"
        report_content += f"- Google Drive Columns: {len(self.analysis['google_drive_columns'])}\n"
        report_content += f"- Timestamp Columns: {len(self.analysis['response_timestamps'])}\n"
        
        # Add image analysis if available
        if ai_analysis and 'individual_analyses' in ai_analysis:
            report_content += "\n### 🤖 AI Image Analysis Summary\n"
            
            sentiments = []
            categories = []
            
            for analysis in ai_analysis['individual_analyses'].values():
                ai_info = analysis.get('ai_analysis', {})
                if ai_info.get('sentiment'):
                    sentiments.append(ai_info['sentiment'])
                if ai_info.get('categories'):
                    categories.extend(ai_info['categories'])
            
            if sentiments:
                sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
                report_content += "\n**Sentiment Distribution:**\n"
                for sentiment, count in sentiment_counts.items():
                    pct = (count / len(sentiments)) * 100
                    report_content += f"- {sentiment.title()}: {count} ({pct:.1f}%)\n"
            
            if categories:
                category_counts = {c: categories.count(c) for c in set(categories)}
                report_content += "\n**Top Content Categories:**\n"
                for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    report_content += f"- {category.title()}: {count} mentions\n"
        
        # Add recommendations
        report_content += "\n## 💡 Recommendations\n"
        
        if drive_analysis and drive_analysis.get('recommendations'):
            report_content += "\n**Google Drive Access:**\n"
            for rec in drive_analysis['recommendations']:
                report_content += f"- {rec}\n"
        
        if self.analysis['data_quality']['missing_data_percentage'] > 20:
            report_content += "- Consider improving data collection to reduce missing responses\n"
        
        if downloaded_images and drive_analysis:
            if len(downloaded_images) < drive_analysis.get('total_links', 0):
                report_content += "- Some images could not be downloaded - check sharing permissions\n"
        
        report_content += "\n## 📈 Next Steps\n"
        report_content += "- Review AI analysis insights for actionable findings\n"
        report_content += "- Consider follow-up surveys based on image analysis results\n"
        report_content += "- Use sentiment analysis to improve future survey design\n"
        
        # Add AI text insights if available
        if ai_text_insights:
            report_content += "\n### 📝 AI Insights for Open-Ended Responses\n"
            for q, ins in ai_text_insights.items():
                report_content += f"\n#### {q}\n"
                report_content += f"\n**Summary:** {ins.get('summary', '')}\n"
                if ins.get('actionable_insights'):
                    report_content += "\n**Actionable Insights:**\n"
                    for a in ins['actionable_insights']:
                        report_content += f"- {a}\n"
        
        # Add VC trend insights if available
        if vc_trend_insights:
            report_content += "\n## 📈 VC Trend Insights\n" + vc_trend_insights + "\n"
        
        report_content += f"\n---\n*Report generated by Comprehensive Survey Analyzer v1.0*"
        
        return report_content
    
    def generate_summary_stats(self) -> Dict:
        """Generate summary statistics for the report"""
        stats = {
            "total_responses": len(self.survey_data),
            "completion_rate": self.analysis['data_quality']['completion_rate'],
            "missing_data": self.analysis['data_quality']['missing_data_percentage'],
            "duplicates": self.analysis['data_quality']['duplicate_responses'],
            "column_types": {
                "text": len(self.analysis['text_columns']),
                "numeric": len(self.analysis['numeric_columns']),
                "choice": len(self.analysis['choice_columns']),
                "email": len(self.analysis['email_columns']),
                "zip": len(self.analysis['zip_columns']),
                "timestamp": len(self.analysis['response_timestamps']),
                "google_drive": len(self.analysis['google_drive_columns'])
            }
        }
        
        # Add quality score
        quality_score = (
            stats['completion_rate'] - 
            (stats['missing_data'] * 0.5) - 
            (stats['duplicates'] * 2)
        )
        stats['quality_score'] = max(0, min(100, quality_score))
        
        return stats 