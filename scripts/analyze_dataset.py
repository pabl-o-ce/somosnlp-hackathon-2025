#!/usr/bin/env python3
"""
Script to identify and filter dataset examples that exceed token length limits.
"""

from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd

def convert_conversation_to_text(conversation):
    """
    Convert a conversation list to a single text string.
    
    Args:
        conversation (list): List of message dictionaries with 'role' and 'content'
    
    Returns:
        str: Formatted conversation text
    """
    if not isinstance(conversation, list):
        return str(conversation) if conversation is not None else ""
    
    text_parts = []
    for message in conversation:
        if isinstance(message, dict) and 'content' in message and 'role' in message:
            role = message['role']
            content = message['content']
            
            # Format based on role (you can customize this format)
            if role == 'system':
                text_parts.append(f"System: {content}")
            elif role == 'user':
                text_parts.append(f"User: {content}")
            elif role == 'assistant':
                text_parts.append(f"Assistant: {content}")
            else:
                text_parts.append(f"{role.title()}: {content}")
    
    return "\n".join(text_parts)

def analyze_long_sequences(dataset_name, max_length=2048, tokenizer_name="microsoft/DialoGPT-medium"):
    """
    Identify and analyze examples that exceed the specified token length.
    
    Args:
        dataset_name (str): HuggingFace dataset identifier
        max_length (int): Maximum token length threshold
        tokenizer_name (str): Tokenizer to use for length calculation
    """
    
    print(f"Loading dataset: {dataset_name}")
    print(f"Using tokenizer: {tokenizer_name}")
    print(f"Analyzing sequences longer than {max_length} tokens")
    print("-" * 60)
    
    # Load dataset and tokenizer
    dataset = load_dataset(dataset_name)
    split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
    data = dataset[split_name]
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Total examples in dataset: {len(data)}")
    
    # Inspect first few examples for data quality
    print(f"\nDATA QUALITY CHECK:")
    print("-" * 30)
    none_chosen = sum(1 for ex in data[:100] if ex.get('chosen') is None)
    none_rejected = sum(1 for ex in data[:100] if ex.get('rejected') is None)
    empty_chosen = sum(1 for ex in data[:100] if not convert_conversation_to_text(ex.get('chosen')).strip())
    empty_rejected = sum(1 for ex in data[:100] if not convert_conversation_to_text(ex.get('rejected')).strip())
    
    print(f"In first 100 examples:")
    print(f"  None 'chosen' values: {none_chosen}")
    print(f"  None 'rejected' values: {none_rejected}")
    print(f"  Empty 'chosen' conversations: {empty_chosen}")
    print(f"  Empty 'rejected' conversations: {empty_rejected}")
    
    # Show data types and conversation structure
    first_example = data[0]
    print(f"\nFirst example data types:")
    print(f"  'chosen' type: {type(first_example.get('chosen'))}")
    print(f"  'rejected' type: {type(first_example.get('rejected'))}")
    
    if first_example.get('chosen') is not None:
        chosen_convo = first_example['chosen']
        if isinstance(chosen_convo, list) and len(chosen_convo) > 0:
            print(f"  'chosen' conversation length: {len(chosen_convo)} messages")
            print(f"  'chosen' first message: {chosen_convo[0]}")
        chosen_text = convert_conversation_to_text(chosen_convo)
        chosen_preview = chosen_text[:200] + "..." if len(chosen_text) > 200 else chosen_text
        print(f"  'chosen' as text preview: {chosen_preview}")
    
    if first_example.get('rejected') is not None:
        rejected_convo = first_example['rejected']
        if isinstance(rejected_convo, list) and len(rejected_convo) > 0:
            print(f"  'rejected' conversation length: {len(rejected_convo)} messages")
        rejected_text = convert_conversation_to_text(rejected_convo)
        rejected_preview = rejected_text[:200] + "..." if len(rejected_text) > 200 else rejected_text
        print(f"  'rejected' as text preview: {rejected_preview}")
    
    print("\n" + "-" * 60)
    
    # Analyze each example
    long_examples = []
    chosen_lengths = []
    rejected_lengths = []
    skipped_examples = 0
    
    for i, example in enumerate(data):
        # Convert conversation lists to text
        chosen_text = convert_conversation_to_text(example.get('chosen'))
        rejected_text = convert_conversation_to_text(example.get('rejected'))
        
        # Skip empty examples
        if not chosen_text.strip() and not rejected_text.strip():
            print(f"Warning: Example {i} has empty chosen and rejected fields")
            skipped_examples += 1
            continue
            
        chosen_tokens = len(tokenizer.encode(chosen_text, add_special_tokens=True))
        rejected_tokens = len(tokenizer.encode(rejected_text, add_special_tokens=True))
        
        chosen_lengths.append(chosen_tokens)
        rejected_lengths.append(rejected_tokens)
        
        # Check if either chosen or rejected exceeds max_length
        if chosen_tokens > max_length or rejected_tokens > max_length:
            long_examples.append({
                'index': i,
                'chosen_length': chosen_tokens,
                'rejected_length': rejected_tokens,
                'chosen_text': chosen_text,
                'rejected_text': rejected_text,
                'exceeds_by_chosen': max(0, chosen_tokens - max_length),
                'exceeds_by_rejected': max(0, rejected_tokens - max_length)
            })
    
    # Statistics
    total_long = len(long_examples)
    percentage_long = (total_long / len(data)) * 100
    
    print(f"\nRESULTS:")
    print(f"Examples analyzed: {len(data) - skipped_examples}")
    print(f"Examples skipped (empty): {skipped_examples}")
    print(f"Examples exceeding {max_length} tokens: {total_long} ({percentage_long:.2f}%)")
    print(f"Examples within limit: {len(data) - total_long - skipped_examples} ({100 - percentage_long:.2f}%)")
    
    if total_long > 0:
        # Analyze the long examples
        chosen_exceeds = sum(1 for ex in long_examples if ex['chosen_length'] > max_length)
        rejected_exceeds = sum(1 for ex in long_examples if ex['rejected_length'] > max_length)
        both_exceed = sum(1 for ex in long_examples if ex['chosen_length'] > max_length and ex['rejected_length'] > max_length)
        
        print(f"\nBREAKDOWN:")
        print(f"'chosen' field exceeds limit: {chosen_exceeds} examples")
        print(f"'rejected' field exceeds limit: {rejected_exceeds} examples")
        print(f"Both fields exceed limit: {both_exceed} examples")
        
        # Show statistics of long examples
        long_chosen_lengths = [ex['chosen_length'] for ex in long_examples if ex['chosen_length'] > max_length]
        long_rejected_lengths = [ex['rejected_length'] for ex in long_examples if ex['rejected_length'] > max_length]
        
        if long_chosen_lengths:
            print(f"\nLONG 'CHOSEN' EXAMPLES:")
            print(f"  Count: {len(long_chosen_lengths)}")
            print(f"  Max length: {max(long_chosen_lengths)}")
            print(f"  Min length: {min(long_chosen_lengths)}")
            print(f"  Average length: {sum(long_chosen_lengths) / len(long_chosen_lengths):.1f}")
        
        if long_rejected_lengths:
            print(f"\nLONG 'REJECTED' EXAMPLES:")
            print(f"  Count: {len(long_rejected_lengths)}")
            print(f"  Max length: {max(long_rejected_lengths)}")
            print(f"  Min length: {min(long_rejected_lengths)}")
            print(f"  Average length: {sum(long_rejected_lengths) / len(long_rejected_lengths):.1f}")
        
        # Show some examples
        print(f"\nSAMPLE LONG EXAMPLES:")
        print("=" * 60)
        for i, ex in enumerate(long_examples[:5]):  # Show first 5 long examples
            print(f"\nExample {i+1} (Index {ex['index']}):")
            print(f"Chosen length: {ex['chosen_length']} tokens")
            print(f"Rejected length: {ex['rejected_length']} tokens")
            
            # Show first 200 characters of chosen text
            chosen_preview = ex['chosen_text'][:200] + "..." if len(ex['chosen_text']) > 200 else ex['chosen_text']
            print(f"Chosen text preview: {chosen_preview}")
            
            # Show first 200 characters of rejected text
            rejected_preview = ex['rejected_text'][:200] + "..." if len(ex['rejected_text']) > 200 else ex['rejected_text']
            print(f"Rejected text preview: {rejected_preview}")
            print("-" * 40)
        
        if len(long_examples) > 5:
            print(f"... and {len(long_examples) - 5} more examples")
    
    # Create filtered dataset
    print(f"\nCREATING FILTERED DATASET:")
    print("=" * 60)
    
    def filter_function(example):
        # Convert conversation lists to text
        chosen_text = convert_conversation_to_text(example.get('chosen'))
        rejected_text = convert_conversation_to_text(example.get('rejected'))
        
        # Skip empty examples
        if not chosen_text.strip() and not rejected_text.strip():
            return False
            
        chosen_length = len(tokenizer.encode(chosen_text, add_special_tokens=True))
        rejected_length = len(tokenizer.encode(rejected_text, add_special_tokens=True))
        return chosen_length <= max_length and rejected_length <= max_length
    
    filtered_data = data.filter(filter_function)
    
    print(f"Original dataset size: {len(data)}")
    print(f"Filtered dataset size: {len(filtered_data)}")
    print(f"Removed examples: {len(data) - len(filtered_data)}")
    print(f"Retention rate: {(len(filtered_data) / len(data)) * 100:.2f}%")
    
    # Save filtered dataset
    print(f"\nSaving filtered dataset...")
    filtered_dataset = dataset
    filtered_dataset[split_name] = filtered_data
    
    # Option 1: Save to local directory
    filtered_dataset.save_to_disk("./filtered_dataset")
    print(f"Filtered dataset saved to './filtered_dataset'")
    
    # Option 2: Save as CSV for inspection
    filtered_df = filtered_data.to_pandas()
    filtered_df.to_csv("filtered_dataset.csv", index=False)
    print(f"Filtered dataset also saved as 'filtered_dataset.csv'")
    
    return filtered_data, long_examples

def examine_specific_long_example(dataset_name, example_index, tokenizer_name="microsoft/DialoGPT-medium"):
    """
    Examine a specific example in detail.
    """
    dataset = load_dataset(dataset_name)
    split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
    data = dataset[split_name]
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    example = data[example_index]
    
    # Convert conversation lists to text
    chosen_text = convert_conversation_to_text(example.get('chosen'))
    rejected_text = convert_conversation_to_text(example.get('rejected'))
    
    print(f"DETAILED ANALYSIS OF EXAMPLE {example_index}:")
    print("=" * 60)
    
    chosen_tokens = tokenizer.encode(chosen_text, add_special_tokens=True)
    rejected_tokens = tokenizer.encode(rejected_text, add_special_tokens=True)
    
    print(f"Chosen length: {len(chosen_tokens)} tokens")
    print(f"Rejected length: {len(rejected_tokens)} tokens")
    
    print(f"\nChosen text:")
    print("-" * 30)
    print(chosen_text)
    
    print(f"\nRejected text:")
    print("-" * 30)
    print(rejected_text)

if __name__ == "__main__":
    dataset_name = "somosnlp-hackathon-2025/gastronomia-hispana-dpo"
    max_length = 2048
    tokenizer_name = "microsoft/DialoGPT-medium"
    
    # Analyze long sequences
    filtered_data, long_examples = analyze_long_sequences(dataset_name, max_length, tokenizer_name)
    
    # Optionally examine specific examples
    # Uncomment the lines below to examine specific long examples:
    # if long_examples:
    #     print("\n" + "="*60)
    #     examine_specific_long_example(dataset_name, long_examples[0]['index'], tokenizer_name)
    
    print(f"\nTo load the filtered dataset later:")
    print(f"from datasets import load_from_disk")
    print(f"filtered_dataset = load_from_disk('./filtered_dataset')")