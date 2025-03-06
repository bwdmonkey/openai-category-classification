# openai-category-classification
OpenAI based document category classification

Supported Document Types:
- png, jpeg, jpg, webp, gif, bmp
- csv, xls, xlsx

Future Document Types:
- doc, docx

## Setup

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Set your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY=your-api-key-here  # On Windows: set OPENAI_API_KEY=your-api-key-here
   ```

## Usage

### Classify a Single Image

```bash
python -m scripts.classify single data/image.jpg
```

### Batch Classify Images in a Directory

```bash
python -m scripts.classify batch data/ --output results.json --categories "Category1" "Category2" "Category3"
```
