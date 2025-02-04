
from typing import List, Dict, Tuple, Optional
import numpy as np
from PyPDF2 import PdfReader
import groq
import os
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import time
from math import ceil
import concurrent.futures
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import re
import time
from dotenv import load_dotenv 

from typing import List

#These ar the system prompts given to control the output of llms
prompt_for_chunk_relevance="""
                        You are an expert in evaluating information relevance for heavy industries like oil, steel, and cemen.You are very less talkative and less freindly thus you only do what is told to you. Your task is to assess whether the given text contains important tender information related to any of the following key elements:

                        1.  Technical Specifications and Requirements 
                          - Detailed quality parameters and industry-specific standards.
                          - Required certifications and compliance requirements.
                          - Technical performance specifications.
                          - Testing and inspection requirements.

                        2.  Commercial Terms 
                          - Pricing structure and payment terms.
                          - Delivery schedules and locations.
                          - Quantity requirements and tolerance levels.
                          - Contract duration and renewal terms.
                          - Performance guarantees and warranties.

                        3.  Legal and Regulatory Requirements 
                          - Industry-specific permits and licenses.
                          - Environmental compliance requirements.
                          - Safety standards and regulations.
                          - Local content requirements.
                          - Insurance and liability coverage.

                        4.  Vendor Qualification Criteria 
                          - Financial capability requirements.
                          - Past experience and track record.
                          - Required certifications and accreditations.
                          - Equipment and facility requirements.

                        5.  Project-Specific Information 
                          - Scope of work/supply.
                          - Project timeline.
                          - Site conditions (if applicable).
                          - Storage and handling requirements.

                        6.  Quality Control and Assurance 
                          - Quality management system requirements.
                          - Inspection and testing procedures.
                          - Sampling and testing standards.
                          - Documentation requirements.

                        7.  Risk Management 
                          - Force majeure clauses.
                          - Penalty clauses.
                          - Performance security requirements.
                          - Dispute resolution mechanisms.

                        8.  Documentation Requirements 
                          - Required certificates and test reports.
                          - Manufacturing data records.
                          - Shipping and packaging documentation.
                          - Compliance certificates.

                            Instructions:
                        - Return only a JSON object with two fields:
                          - `"is_relevant"`: A boolean value indicating whether the text is relevant.
                          - `"relevance_score"`: A float value between 0 and 1 indicating the degree of relevance (higher is better).

                        STRICTLY ONLY GIVE JSON FILE THE WAY I SUGGESTED NO OTHER TEXT OR INFO  OR ANY OTHER THING.

                        Example Response:
                        ```json
                        {
                          "is_relevant": true,
                          "relevance_score": 0.85
                        }

                        STRICTLY ONLY GIVE JSON FILE THE WAY I SUGGESTED NO OTHER TEXT OR INFO  OR ANY OTHER THING.
                                """
prompt_for_summarization="""
                        You are an expert in summarizing tender documents for heavy industries like oil, steel, and cement. Your task is to provide concise, accurate summaries of tender document sections while retaining key information relevant.

                        Key Elements to Retain if present any:

                        1. Technical Specifications and Requirements
                          - Detailed quality parameters and industry-specific standards.
                          - Required certifications and compliance requirements.
                          - Technical performance specifications.
                          - Testing and inspection requirements.

                        2. Commercial Terms
                          - Pricing structure and payment terms.
                          - Delivery schedules and locations.
                          - Quantity requirements and tolerance levels.
                          - Contract duration and renewal terms.
                          - Performance guarantees and warranties.

                        3. Legal and Regulatory Requirements
                          - Industry-specific permits and licenses.
                          - Environmental compliance requirements.
                          - Safety standards and regulations.
                          - Local content requirements.
                          - Insurance and liability coverage.

                        4. Vendor Qualification Criteria
                          - Financial capability requirements.
                          - Past experience and track record.
                          - Required certifications and accreditations.
                          - Equipment and facility requirements.

                        5. Project-Specific Information
                          - Scope of work or supply.
                          - Project timeline.
                          - Site conditions (if applicable).
                          - Storage and handling requirements.

                        6. Quality Control and Assurance
                          - Quality management system requirements.
                          - Inspection and testing procedures.
                          - Sampling and testing standards.
                          - Documentation requirements.

                        7. Risk Management
                          - Force majeure clauses.
                          - Penalty clauses.
                          - Performance security requirements.
                          - Dispute resolution mechanisms.

                        8. Documentation Requirements
                          - Required certificates and test reports.
                          - Manufacturing data records.
                          - Shipping and packaging documentation.
                          - Compliance certificates.

                            Instructions:
                        - Summarize the section in clear and concise language that retains all critical elements.
                        - Return the summary as plain text.
                        """
prompt_for_final_summarization="""
                        You are an expert in summarizing tender documents for heavy industries like oil, steel, and cement. Your task is to provide concise, accurate summaries of tender document in the format given below sections while retaining key information relevant.

                        Key Elements to Retain if present any:

                        1. Technical Specifications and Requirements
                          - Detailed quality parameters and industry-specific standards.
                          - Required certifications and compliance requirements.
                          - Technical performance specifications.
                          - Testing and inspection requirements.

                        2. Commercial Terms
                          - Pricing structure and payment terms.
                          - Delivery schedules and locations.
                          - Quantity requirements and tolerance levels.
                          - Contract duration and renewal terms.
                          - Performance guarantees and warranties.

                        3. Legal and Regulatory Requirements
                          - Industry-specific permits and licenses.
                          - Environmental compliance requirements.
                          - Safety standards and regulations.
                          - Local content requirements.
                          - Insurance and liability coverage.

                        4. Vendor Qualification Criteria
                          - Financial capability requirements.
                          - Past experience and track record.
                          - Required certifications and accreditations.
                          - Equipment and facility requirements.

                        5. Project-Specific Information
                          - Scope of work or supply.
                          - Project timeline.
                          - Site conditions (if applicable).
                          - Storage and handling requirements.

                        6. Quality Control and Assurance
                          - Quality management system requirements.
                          - Inspection and testing procedures.
                          - Sampling and testing standards.
                          - Documentation requirements.

                        7. Risk Management
                          - Force majeure clauses.
                          - Penalty clauses.
                          - Performance security requirements.
                          - Dispute resolution mechanisms.

                        8. Documentation Requirements
                          - Required certificates and test reports.
                          - Manufacturing data records.
                          - Shipping and packaging documentation.
                          - Compliance certificates.

                            Instructions:
                        - Summarize the section in clear and concise language that retains all critical elements.
                        - Return the summary as plain text.  this is the format , give summary this way.
                        """



@dataclass # NOt used here extensively but can be used for advanced filtering purpose
class ChunkInfo:
    text: str
    length: int
    relevance_score: float = 0.0
    summary: str = ""


''' 
    Here in the program multiple api keys because 
    - To make system robust by each api called round robin wise thus none of them would exeed rate limit
    - We want to make system free of use for larger pdfs. 
 '''
 
class TenderSummarizer:
    def __init__(self, api_keys: List[str], relevance_api_count: int = 4):
        if len(api_keys) < relevance_api_count:
            raise ValueError(f"Need at least {relevance_api_count} API keys")

        self.relevance_api_keys = api_keys  # These api keys of groq cloud llms for relevant chunks filtering
        self.summary_api_keys = api_keys    # 

        self.relevance_clients = [groq.Groq(api_key=key) for key in self.relevance_api_keys]
        self.summary_clients = [groq.Groq(api_key=key) for key in self.summary_api_keys]

        self.current_relevance_key_index = 0
        self.current_summary_key_index = 0

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _get_next_client(self, for_relevance: bool = True) -> groq.Groq:
          time.sleep(0.1)  # Add small delay between API calls
          if for_relevance:
              client = self.relevance_clients[self.current_relevance_key_index]
              self.current_relevance_key_index = (self.current_relevance_key_index + 1) % len(self.relevance_clients)
          else:
              client = self.summary_clients[self.current_summary_key_index]
              self.current_summary_key_index = (self.current_summary_key_index + 1) % len(self.summary_clients)
          return client

    def merge_chunks(self, chunks: List[ChunkInfo], target_size: int = 4000) -> List[ChunkInfo]:
            """Merge adjacent chunks to reach target size while handling edge cases."""
            if not chunks:
                return []

            merged_chunks = []
            current_chunks = []
            current_length = 0

            for chunk in chunks:
                if current_length + chunk.length > target_size and current_chunks:
                    # If we would exceed target size, create a new merged chunk
                    merged_text = ' '.join(c.text for c in current_chunks)
                    merged_score = sum(c.relevance_score for c in current_chunks) / len(current_chunks)
                    merged_chunks.append(ChunkInfo(
                        text=merged_text,
                        length=current_length,
                        relevance_score=merged_score
                    ))
                    current_chunks = []
                    current_length = 0

                current_chunks.append(chunk)
                current_length += chunk.length

            # Handle remaining chunks
            if current_chunks:
                merged_text = ' '.join(c.text for c in current_chunks)
                merged_score = sum(c.relevance_score for c in current_chunks) / len(current_chunks)
                merged_chunks.append(ChunkInfo(
                    text=merged_text,
                    length=current_length,
                    relevance_score=merged_score
                ))

            return merged_chunks


    def extract_text(self, pdf_path: str) -> str:

          with open(pdf_path, 'rb') as file:
              reader = PdfReader(file)
              text = ""
              for page in reader.pages:
                  text += page.extract_text()
          return text


    def create_initial_chunks(self, text: str, target_size: int = 200, overlap: int = 20) -> List[ChunkInfo]:
            """
            Create linear chunks of text with a specified target size and overlap.

            Args:
                text (str): The input text to split into chunks.
                target_size (int): The target size (in tokens) for each chunk.
                overlap (int): The number of overlapping tokens between consecutive chunks.

            Returns:
                List[ChunkInfo]: A list of ChunkInfo objects containing text and token length.
            """
            words = text.split()  # Split text into words (tokens)
            chunks = []

            start = 0
            while start < len(words):
                # Define the end index of the current chunk
                end = min(start + target_size, len(words))

                # Create the chunk text and calculate its length
                chunk_text = " ".join(words[start:end])
                chunk_length = len(words[start:end])

                # Add the chunk to the list
                chunks.append(ChunkInfo(text=chunk_text, length=chunk_length))

                # Update the start index for the next chunk, considering overlap
                start += target_size - overlap

            return chunks


    def extract_json_from_string(self ,input_string: str):
            """
            Extract JSON data from a string that may contain additional text.

            Args:
                input_string (str): The input string containing JSON and potentially extra text.

            Returns:
                dict: The extracted JSON data as a Python dictionary.
                str: Error message if JSON extraction fails.
            """
            try:
                # Match the first JSON-like structure in the string using regex
                json_pattern = r'(\{.*?\}|\[.*?\])'
                match = re.search(json_pattern, input_string, re.DOTALL)

                if match:
                    # Parse the JSON portion of the string
                    json_data = json.loads(match.group(0))
                    return json_data
                else:
                    return {"error": "No valid JSON found in the input string."}
            except json.JSONDecodeError as e:
                return {"error": f"Invalid JSON: {str(e)}"}

    def assess_chunk_relevance(self, chunk: ChunkInfo,api_key) -> Tuple[bool, float]:
            system_prompt = prompt_for_chunk_relevance
            
            client = groq.Groq(api_key=api_key)

                    # Add timeout parameter
            response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": chunk.text}
                        ],
                        temperature=0.1,
                        max_tokens=100,
                                       # Add timeout of 30 seconds
                    )
            
                    # Extract JSON data from the response
            json_data = self.extract_json_from_string(response.choices[0].message.content)
           
            if "error" in json_data:
                        logging.error(f"Error in JSON extraction: {json_data['error']}")
                        return False, 0.0

            return json_data.get("is_relevant", False), json_data.get("relevance_score", 0.0)


    def summarize_chunk(self, chunk: ChunkInfo) -> str:

        system_prompt = prompt_for_summarization

        try:
            client = self._get_next_client(for_relevance=False)
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk.text}
                ],
                temperature=1.0,
                max_tokens=100
            )

            summary_tmp = response.choices[0].message.content
            formatted_summary = "\n".join(summary_tmp.split(". "))
            return formatted_summary
            # return response.choices[0].message.content
        
        except Exception as e:
            logging.error(f"Error in chunk summarization: {str(e)}")
            return ""

    def create_final_summary(self, summaries: List[str],api_key: str) -> str:

        """Create final summary from intermediate summaries."""

        combined_text = " ".join(summaries)

        system_prompt = prompt_for_final_summarization

        try:
            client = groq.Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-specdec",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": combined_text}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            final_summary = response.choices[0].message.content
            return final_summary
            # return response.choices[0].message.content
        
        except Exception as e:
            logging.error(f"Error in final summarization: {str(e)}")
            return ""


    def process_tender_document(self, pdf_path: str) -> str:
        """Process tender document without using Dask."""
        try:
            # Extract text
            logging.info("Extracting text from PDF...")
            text = self.extract_text(pdf_path)
            
            api_key = self.summary_api_keys[0]

            # Handle very short documents
            if len(text.split()) < 500:
                logging.info("Document is very short, creating direct summary...")
                return self.create_final_summary([text], api_key)

            # Create initial chunks
            logging.info("Creating initial chunks...")
            initial_chunks = self.create_initial_chunks(text, target_size=300)
            print("length of initial chunks" ,len(initial_chunks))

            # Assess relevance sequentially
            logging.info("Assessing chunk relevance...")
            relevance_results = []
            

            # Iterate over chunks and process them one at a time
            for i, chunk in enumerate(initial_chunks):       
                          api_key = self.relevance_api_keys[i % len(self.relevance_api_keys)]         
                          result = self.assess_chunk_relevance(chunk, api_key)       
                          relevance_results.append(result)


            # Filter and score relevant chunks
            relevant_chunks = []
            for chunk, (is_relevant, score) in zip(initial_chunks, relevance_results):
                if is_relevant:
                    chunk.relevance_score = score
                    relevant_chunks.append(chunk)

            logging.info(f"Found {len(relevant_chunks)} relevant chunks")

            if not relevant_chunks:
                logging.warning("No relevant chunks found, using all chunks...")
                relevant_chunks = initial_chunks

       

            # Merge chunks
            current_chunks = self.merge_chunks(relevant_chunks, target_size=2000)
            total_tokens = sum(chunk.length for chunk in current_chunks)

            print("current chunks length", len(current_chunks))

            # Progressive summarization
            while total_tokens > 2000:
                    logging.info(f"Current total tokens: {total_tokens}")

                    # Summarize chunks sequentially
                    summaries = []
                    for chunk in current_chunks:
                        summary = self.summarize_chunk(chunk)
                        if summary:
                            summaries.append(summary)

                    # Prepare for next iteration
                    current_chunks = [
                        ChunkInfo(text=summary, length=int(len(summary.split()) * 1.3))
                        for summary in summaries
                    ]

                    # Merge chunks if needed
                    if len(current_chunks) > 1:
                        current_chunks = self.merge_chunks(current_chunks, target_size=4000)

                    total_tokens = sum(chunk.length for chunk in current_chunks)


           
            # Create final summary
            logging.info("Creating final summary...")

            api_key = self.summary_api_keys[0]
            final_summary = self.create_final_summary([chunk.text for chunk in current_chunks],api_key)

            return final_summary

        except Exception as e:
            logging.error(f"Error processing document: {str(e)}")
            raise


def summarize_pdf(pdf_path: str) -> str:
    load_dotenv()  # Call this ONCE, ideally at the beginning of your script

    # Get API keys from environment variables
    api_keys = []
    for i in range(1, 6):  # Assuming you have 5 keys named API_KEY_1 to API_KEY_5
        key = os.getenv(f"API_KEY_{i}")
        if key:
            api_keys.append(key)
        else:
            logging.warning(f"API_KEY_{i} not found in .env file.")

    if not api_keys:
        raise ValueError("No API keys found in .env file.  Please set API_KEY_1, API_KEY_2, etc.")

    summarizer = TenderSummarizer(api_keys, relevance_api_count=4)

    try:
        summary = summarizer.process_tender_document(pdf_path)
        return summary
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")
        return str(e)


