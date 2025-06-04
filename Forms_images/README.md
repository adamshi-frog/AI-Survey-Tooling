# Comprehensive Survey Analyzer

A powerful Streamlit application for analyzing Google Forms survey responses, with advanced features for data visualization, AI-powered insights, and image analysis.

## Features

- 📊 **Data Analysis**
  - Upload and analyze Google Forms CSV exports
  - Automatic column type detection
  - Data quality metrics and visualization
  - Response timeline analysis
  - Text response analysis with word frequency

- 🤖 **AI-Powered Insights**
  - OpenAI integration for text analysis
  - AI-generated insights for open-ended questions
  - Venture Capital trend analysis (Beta)
  - Customizable analysis prompts

- 🖼️ **Image Analysis**
  - Google Drive image download and processing
  - Image metadata extraction
  - AI-powered image analysis
  - Image quality assessment

- 📋 **Reporting**
  - Comprehensive analysis reports
  - Multiple export formats (Markdown, CSV, JSON)
  - Complete analysis package downloads
  - Customizable report generation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI-Survey-Tooling.git
cd AI-Survey-Tooling/Forms_images
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run main.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Upload your Google Forms CSV export or use sample data

4. Configure analysis options in the sidebar

5. Explore the analysis results in the different tabs

## Project Structure

```
Forms_images/
├── main.py                 # Main application entry point
├── config.py              # Configuration settings
├── requirements.txt       # Project dependencies
├── core/                  # Core analysis modules
│   ├── analyzer.py       # Survey data analysis
│   ├── image_processor.py # Image processing
│   └── report_generator.py # Report generation
├── ui/                    # UI components
│   ├── components.py     # Reusable UI components
│   └── styling.py        # Custom styling
└── utils/                 # Utility modules
    ├── file_utils.py     # File handling utilities
    ├── logging.py        # Logging utilities
    └── openai_utils.py   # OpenAI API utilities
```

## Configuration

The application can be configured through the `config.py` file:

- `APP_TITLE`: Application title
- `APP_DESCRIPTION`: Application description
- `DEFAULT_OUTPUT_DIR`: Default directory for downloaded files
- `OPENAI_API_KEY`: OpenAI API key
- `ENABLE_AI_BY_DEFAULT`: Enable AI analysis by default
- `SHOW_API_KEY_INPUT`: Show API key input field
- `MAX_SAMPLE_RESPONSES`: Maximum number of responses to sample
- `MAX_IMAGE_DIMENSION`: Maximum image dimension for processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Streamlit for the amazing web application framework
- OpenAI for the powerful AI capabilities
- Google Forms for the survey platform 