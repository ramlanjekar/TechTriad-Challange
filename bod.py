
from PyPDF2 import PdfReader
import pandas as pd

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



@dataclass
class BOQItem:
    product_name: str
    specification: str
    quantity: str
    unit: str
    page_number: int = 0

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
    def __init__(self):
        """Initialize with Groq client."""
        # Get first available API key from environment
        api_key = userdata.get('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")

        self.client = groq.Groq(api_key=api_key)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def extract_text(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from PDF and return dict with page numbers and text."""
        page_texts = {}

        try:
            with open(pdf_path, 'rb') as file:
                pdf = PdfReader(file)
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text.strip():
                        cleaned_text = " ".join(line.strip() for line in text.splitlines())
                        page_texts[i] = cleaned_text
        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            raise

        return page_texts

    def extract_fields(self, chunk: str) -> List[BOQItem]:
        """Extract BOQ fields with validation and cleaning."""
        prompt = """You are a precise extractor that only outputs valid JSON arrays. Extract these fields from the text:
        1. Product Name
        2. Product Specification with Description
        3. Quantity
        4. Unit of Measurement

        Only output a valid JSON array like this, with no additional text:
        [
            {
                "product_name": "example name",
                "specification": "detailed spec",
                "quantity": "numeric value",
                "unit": "unit of measurement"
            }
        ]

        If no items are found, output an empty array: []"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": chunk}
                ],
                temperature=0.2,  # Lower temperature for more consistent output
                max_tokens=1000
            )

            content = response.choices[0].message.content

            # Debug logging
            self.logger.debug(f"Raw LLM response: {content}")

            # Clean the content to ensure it's valid JSON
            content = content.strip()
            # Remove any markdown code block markers if present
            content = content.replace('```json', '').replace('```', '')

            try:
                items = json.loads(content) if content else []
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error: {e}")
                self.logger.error(f"Problematic content: {content}")
                return []

            if not isinstance(items, list):
                self.logger.error(f"Expected list but got {type(items)}")
                return []

            boq_items = []
            for item in items:
                try:
                    boq_item = BOQItem(**item)
                    boq_item.clean()
                    is_valid, errors = boq_item.validate()
                    if is_valid:
                        boq_items.append(boq_item)

                except Exception as e:
                    self.logger.warning(f"Error creating BOQ item: {e}")
                    continue

            return boq_items

        except Exception as e:
            self.logger.error(f"Error extracting fields: {e}")
            return []

    def process_document(self, pdf_path: str, output_path: str) -> None:
        """Process document and save to Excel."""
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            page_texts = self.extract_text(pdf_path)
            all_items = []

            for page_num, text in page_texts.items():
                self.logger.info(f"Processing page {page_num}...")
                items = self.extract_fields(text)
                if items:  # Only log if items were found
                    self.logger.info(f"Found {len(items)} items on page {page_num}")
                for item in items:
                    item.page_number = page_num
                all_items.extend(items)

            if not all_items:
                self.logger.error("No valid BOQ items extracted from any page")
                raise ValueError("No valid BOQ items extracted")

            df = pd.DataFrame([
                {
                    'Product Name': item.product_name,
                    'Product Specification': item.specification,
                    'Quantity': item.quantity,
                    'Unit': item.unit,
                    'Page Number': item.page_number
                }
                for item in all_items
            ])

            df.to_excel(output_path, index=False)
            self.logger.info(f"Saved {len(df)} items to {output_path}")

        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            raise

def main():
    try:
        extractor = BOQExtractor()
        pdf_path = "/content/tender_601813909e29b_TenderNitPPbag.pdf"
        output_path = "extracted_boq.xlsx"

        extractor.process_document(pdf_path, output_path)

    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
