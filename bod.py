
from PyPDF2 import PdfReader
import pandas as pd
import dask.delayed
import dask.distributed
from typing import List, Dict, Optional, Tuple
import groq
import os
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import re
from google.colab import userdata


@dataclass
class BOQItem:
    product_name: str
    specification: str
    quantity: str
    unit: str
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate BOQ item fields."""
        errors = []
        if not self.product_name.strip():
            errors.append("Empty product name")
        if not self.specification.strip():
            errors.append("Empty specification")
        if not self.quantity.strip():
            errors.append("Empty quantity")
        if not self.unit.strip():
            errors.append("Empty unit")
            
        # Validate quantity format
        try:
            float(self.quantity.replace(',', ''))
        except ValueError:
            errors.append(f"Invalid quantity format: {self.quantity}")
            
        return len(errors) == 0, errors

    def clean(self) -> None:
        """Clean and normalize fields."""
        self.product_name = self.product_name.strip()
        self.specification = self.specification.strip()
        self.quantity = re.sub(r'[^\d.,]', '', self.quantity)
        self.unit = self.unit.strip().lower()

class BOQExtractor:
    def __init__(self, api_keys: List[str], retry_attempts: int = 3, retry_delay: int = 2):
        if not api_keys:
            raise ValueError("No API keys provided")
            
        self.clients = [groq.Groq(api_key=key) for key in api_keys]
        self.current_key_index = 0
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('boq_extraction.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _get_next_client(self) -> groq.Groq:
        """Get next API client with rate limiting."""
        client = self.clients[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.clients)
        time.sleep(0.1)  # Basic rate limiting
        return client

    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF with multiple fallback methods."""
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        if pdf_path.stat().st_size == 0:
            raise ValueError("PDF file is empty")
            
        errors = []
        
        # Try pdfplumber
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                if text.strip():
                    return text
                errors.append("pdfplumber extracted empty text")
        except Exception as e:
            errors.append(f"pdfplumber failed: {str(e)}")
        
        # Try PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                if text.strip():
                    return text
                errors.append("PyPDF2 extracted empty text")
        except Exception as e:
            errors.append(f"PyPDF2 failed: {str(e)}")
            
        raise Exception(f"Failed to extract text from PDF: {'; '.join(errors)}")

    def create_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into chunks with overlap and table preservation."""
        if not text.strip():
            raise ValueError("Empty text provided for chunking")
            
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, line in enumerate(lines):
            line_size = len(line.split())
            
            # Detect potential table rows
            is_table_row = bool(re.search(r'\d+\s*[x×]\s*\d+|[\d.,]+\s*(pcs|m2|m3|kg|tons?)', line, re.I))
            
            if (current_size + line_size > chunk_size and current_chunk and not is_table_row):
                # Add overlap from previous chunk
                overlap_lines = current_chunk[-overlap:] if overlap > 0 else []
                chunks.append('\n'.join(current_chunk))
                current_chunk = overlap_lines.copy()
                current_size = sum(len(l.split()) for l in overlap_lines)
                
            current_chunk.append(line)
            current_size += line_size
            
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        if not chunks:
            raise ValueError("Chunking resulted in no chunks")
            
        return chunks

    def retry_with_exponential_backoff(self, func, *args, **kwargs):
        """Retry function with exponential backoff."""
        for attempt in range(self.retry_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    raise
                wait_time = self.retry_delay * (2 ** attempt)
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

    @dask.delayed
    def extract_fields(self, chunk: str) -> List[BOQItem]:
        """Extract BOQ fields with validation and cleaning."""
        prompt = """Extract these four fields from the text:
        1. Product Name
        2. Product Specification with Description
        3. Quantity (numeric value)
        4. Unit of Measurement
        
        Return as JSON array:
        [
            {
                "product_name": "example name",
                "specification": "detailed spec",
                "quantity": "numeric value",
                "unit": "unit of measurement"
            }
        ]
        
        Guidelines:
        - Only include items with all four fields
        - Clean and normalize quantities to numeric values
        - Standardize units (e.g., 'meter' to 'm')
        - Return empty array if no complete items found"""

        def _extract() -> List[BOQItem]:
            client = self._get_next_client()
            response = client.chat.completions.create(
                model="llama2-70b-4096",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": chunk}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            items = json.loads(response.choices[0].message.content)
            return [BOQItem(**item) for item in items]

        try:
            items = self.retry_with_exponential_backoff(_extract)
            
            # Validate and clean items
            valid_items = []
            for item in items:
                item.clean()
                is_valid, errors = item.validate()
                if is_valid:
                    valid_items.append(item)
                else:
                    self.logger.warning(f"Invalid BOQ item: {errors}")
                    
            return valid_items
            
        except Exception as e:
            self.logger.error(f"Extraction error: {str(e)}")
            return []

    def save_to_excel(self, df: pd.DataFrame, output_path: str) -> None:
        """Save DataFrame to Excel with formatting."""
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='BOQ')
                
                # Auto-adjust column widths
                worksheet = writer.sheets['BOQ']
                for idx, col in enumerate(df.columns):
                    max_length = max(
                        df[col].astype(str).apply(len).max(),
                        len(col)
                    )
                    worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 50)

        except Exception as e:
            raise Exception(f"Failed to save Excel file: {str(e)}")

    def process_document(self, pdf_path: str, output_path: str):
        """Process document with comprehensive error handling."""
        client = None
        try:
            # Validate paths
            pdf_path = Path(pdf_path)
            output_path = Path(output_path)
            
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Setup Dask
            client = dask.distributed.Client()
            
            # Extract and process text
            self.logger.info("Extracting text from PDF...")
            text = self.extract_text(pdf_path)
            
            if not text.strip():
                raise ValueError("Extracted text is empty")
                
            self.logger.info("Creating chunks...")
            chunks = self.create_chunks(text)
            
            # Process chunks in parallel
            self.logger.info("Processing chunks...")
            extraction_tasks = [self.extract_fields(chunk) for chunk in chunks]
            results = dask.compute(*extraction_tasks)
            
            # Flatten and validate results
            all_items = [item for sublist in results for item in sublist]
            
            if not all_items:
                raise ValueError("No valid BOQ items extracted")
                
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'Product Name': item.product_name,
                    'Product Specification': item.specification,
                    'Quantity': item.quantity,
                    'Unit': item.unit
                } 
                for item in all_items
            ])
            
            # Add metadata
            df['Extraction Date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
            df['Source File'] = pdf_path.name
            
            # Save results
            self.logger.info(f"Saving {len(df)} items to Excel...")
            self.save_to_excel(df, output_path)
            
            self.logger.info(f"Successfully processed {pdf_path.name}")
            return df
            
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            raise
            
        finally:
            if client:
                client.close()

def main():
    # Load API keys from environment with validation
    api_keys = [
        key for i in range(1, 11)  # Support up to 10 API keys
        if (key := os.getenv(f"GROQ_API_KEY_{i}"))
    ]
    
    if not api_keys:
        raise ValueError("No API keys found in environment variables")
        
    try:
        extractor = BOQExtractor(api_keys)
        
        pdf_path = "path/to/your/boq.pdf"
        output_path = "extracted_boq.xlsx"
        
        # Process with error handling
        df = extractor.process_document(pdf_path, output_path)
        
        print("\nProcessing Summary:")
        print(f"Total items extracted: {len(df)}")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()