---
name: "markdown-converter"
description: "Converts various document formats (PDF, HTML, DOCX) to Markdown using pymupdf4llm, html2text, and python-docx. Invoke when user wants to convert documents to Markdown format."
---

# Markdown Converter Skill

## Overview
This skill converts various document formats to clean Markdown format using specialized libraries for each input type.

## Supported Formats
- **PDF**: Uses pymupdf4llm for high-quality text extraction and structure preservation
- **HTML**: Uses html2text with BeautifulSoup for clean HTML to Markdown conversion
- **DOCX**: Uses python-docx for direct Word document to Markdown conversion

## Usage

### Basic Conversion
```python
from markdown_converter import convert_to_markdown

# Convert PDF to Markdown
convert_to_markdown("document.pdf", "output.md")

# Convert HTML to Markdown
convert_to_markdown("page.html", "output.md")

# Convert DOCX to Markdown
convert_to_markdown("document.docx", "output.md")
```

### Batch Conversion
```python
from markdown_converter import batch_convert

# Convert all PDFs in directory to Markdown
batch_convert("/path/to/directory", "pdf", "md")
```

## Dependencies
- pymupdf4llm (for PDF conversion)
- html2text (for HTML conversion)
- beautifulsoup4 (for HTML parsing)
- python-docx (for DOCX conversion)
- python-magic (for automatic format detection)

## Installation
```bash
pip install pymupdf4llm html2text beautifulsoup4 python-docx python-magic-bin
```

## Features
- Preserves document structure (headings, lists, tables)
- Handles complex layouts in PDF documents
- Clean HTML stripping and conversion
- Supports batch processing of multiple files
- Automatic format detection