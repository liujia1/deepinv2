from markitdown import convert_file_to_markdown
import os

def convert_to_markdown(input_path, output_path=None):
    """
    Convert input file to Markdown using Microsoft's markitdown library.
    
    Args:
        input_path (str): Path to input file
        output_path (str, optional): Path to output Markdown file. If not provided, will use same directory and name with .md extension.
    
    Returns:
        bool: True if conversion succeeded, False otherwise
    """
    try:
        if not output_path:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = f"{base_name}.md"
            
        convert_file_to_markdown(input_path, output_path)
        print(f"Successfully converted: {input_path} -> {output_path}")
        return True
    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")
        return False

def batch_convert(directory, input_ext=None, output_ext='md'):
    """
    Batch convert files in directory to Markdown using markitdown.
    
    Args:
        directory (str): Path to directory containing files to convert
        input_ext (str, optional): File extension to filter (e.g., '.pdf', '.docx')
        output_ext (str, optional): Output file extension (default: 'md')
    
    Returns:
        int: Number of successful conversions
    """
    success_count = 0
    total_count = 0
    
    for filename in os.listdir(directory):
        if input_ext and not filename.lower().endswith(input_ext):
            continue
            
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            total_count += 1
            print(f"Converting: {filename}")
            
            if convert_to_markdown(file_path):
                success_count += 1
            else:
                print(f"Failed to convert: {filename}")
                
    print(f"\nConversion completed: {success_count}/{total_count} files successful")
    return success_count

# Example usage
if __name__ == "__main__":
    # Convert a single PDF file
    # convert_to_markdown("document.pdf")
    
    # Batch convert all PDF files in directory
    # batch_convert(".", ".pdf")