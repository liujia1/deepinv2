---
name: "markdown-converter"
description: "Converts various document formats (PDF, Word, Excel, PowerPoint, HTML, images, audio) to Markdown using Microsoft's markitdown library. Invoke when user wants to convert documents to Markdown format."
---

# Markdown Converter Skill (Powered by Microsoft MarkItDown)

## Overview
This skill uses Microsoft's official MarkItDown library to convert various document formats to clean Markdown format.

## Supported Formats
- **PDF**: High-quality text extraction with structure preservation
- **Word (DOCX/DOC)**: Direct conversion from Word documents
- **Excel (XLSX/XLS)**: Converts spreadsheets to Markdown tables
- **PowerPoint (PPTX/PPT)**: Extracts slides content to Markdown
- **HTML**: Clean HTML to Markdown conversion
- **Images**: OCR text extraction from images
- **Audio**: Speech transcription to text
- **CSV**: Converts CSV files to Markdown tables
- **JSON/XML**: Structured data to Markdown
- **ZIP**: Batch conversion of files inside ZIP archives

## Usage

### Basic Conversion
```python
from markitdown import convert_file_to_markdown

# Convert PDF to Markdown
convert_file_to_markdown("document.pdf", "output.md")

# Convert Word to Markdown
convert_file_to_markdown("document.docx", "output.md")

# Convert Excel to Markdown
convert_file_to_markdown("spreadsheet.xlsx", "output.md")
```

### Batch Conversion
```python
import os
from markitdown import convert_file_to_markdown

directory = "/path/to/documents"
for filename in os.listdir(directory):
    if filename.lower().endswith(('.pdf', '.docx', '.html')):
        input_path = os.path.join(directory, filename)
        output_path = os.path.join(directory, f"{os.path.splitext(filename)[0]}.md")
        convert_file_to_markdown(input_path, output_path)
```

## Features
- Preserves document structure (headings, lists, tables, links)
- Handles complex layouts in PDF documents
- Supports OCR for image files
- Speech transcription for audio files
- Automatic format detection
- Clean and readable Markdown output

## Installation
```bash
pip install markitdown
```

## Notes
- For large files (>10MB), consider splitting them into smaller chunks
- Image OCR requires internet connection for cloud processing
- Audio transcription may have limitations on file size and language support