import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict, Any

def extract_flattened_data(json_data: List[Dict[Any, Any]]) -> List[Dict[str, Any]]:
    """
    Extract and flatten the nested JSON structure into the desired format.
    """
    flattened_records = []
    
    for record in json_data:
        # Extract system prompt and user question from messages
        system_prompt = ""
        question = ""
        
        for message in record.get("messages", []):
            if message.get("role") == "system":
                system_prompt = message.get("content", "")
            elif message.get("role") == "user":
                question = message.get("content", "")
        
        # Extract metadata fields
        metadata = record.get("metadata", {})
        
        flattened_record = {
            "chosen": record.get("chosen", ""),
            "rejected": record.get("rejected", ""),
            "recipe_id": record.get("recipe_id"),
            "recipe_name": record.get("recipe_name"),
            "category": record.get("category"),
        }
        
        flattened_records.append(flattened_record)
    
    return flattened_records

# Method 1: Load all data in memory (for files that fit in RAM)
def convert_json_to_parquet_pandas(input_file: str, output_file: str):
    """
    Convert JSON to Parquet using pandas (loads all data in memory).
    Best for files that fit comfortably in your 64GB RAM.
    """
    print("Loading JSON file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    print("Flattening data structure...")
    flattened_data = extract_flattened_data(json_data)
    
    print("Creating DataFrame...")
    df = pd.DataFrame(flattened_data)
    
    print("Writing to Parquet...")
    df.to_parquet(output_file, index=False, engine='pyarrow')
    print(f"Successfully converted to {output_file}")
    print(f"Shape: {df.shape}")

# Method 2: Streaming approach for very large files
def convert_json_to_parquet_streaming(input_file: str, output_file: str, chunk_size: int = 1000):
    """
    Convert JSON to Parquet using streaming approach.
    Processes data in chunks to handle very large files.
    """
    print("Loading JSON file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    print("Processing in chunks...")
    
    # Process in chunks
    total_records = len(json_data)
    writer = None
    schema = None
    
    for i in range(0, total_records, chunk_size):
        chunk = json_data[i:i + chunk_size]
        flattened_chunk = extract_flattened_data(chunk)
        df_chunk = pd.DataFrame(flattened_chunk)
        
        # Convert to PyArrow table
        table = pa.Table.from_pandas(df_chunk)
        
        if writer is None:
            # Create writer with schema from first chunk
            schema = table.schema
            writer = pq.ParquetWriter(output_file, schema)
        
        writer.write_table(table)
        print(f"Processed {min(i + chunk_size, total_records)}/{total_records} records")
    
    if writer:
        writer.close()
    
    print(f"Successfully converted to {output_file}")

# Method 3: Using Polars (often faster for large datasets)
def convert_json_to_parquet_polars(input_file: str, output_file: str):
    """
    Convert JSON to Parquet using Polars (requires: pip install polars).
    Often faster than pandas for large datasets.
    """
    try:
        import polars as pl
    except ImportError:
        print("Polars not installed. Install with: pip install polars")
        return
    
    print("Loading JSON file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    print("Flattening data structure...")
    flattened_data = extract_flattened_data(json_data)
    
    print("Creating Polars DataFrame...")
    df = pl.DataFrame(flattened_data)
    
    print("Writing to Parquet...")
    df.write_parquet(output_file)
    print(f"Successfully converted to {output_file}")
    print(f"Shape: {df.shape}")

# Example usage
if __name__ == "__main__":
    input_file = "train-somosnpl-recetas-dpo-v3.json"
    output_file = "train-somosnpl-recetas-dpo-v3.parquet"
    
    # Choose the method that works best for your file size:
    
    # Method 1: Standard pandas approach (recommended for most cases)
    convert_json_to_parquet_pandas(input_file, output_file)
    
    # Method 2: Streaming approach (for very large files)
    # convert_json_to_parquet_streaming(input_file, output_file, chunk_size=1000)
    
    # Method 3: Polars approach (potentially faster)
    # convert_json_to_parquet_polars(input_file, output_file)
    
    # Verify the result
    print("\nVerifying the converted file...")
    df_check = pd.read_parquet(output_file)
    print(f"Final shape: {df_check.shape}")
    print(f"Columns: {list(df_check.columns)}")
    print("\nFirst few rows:")
    print(df_check.head())