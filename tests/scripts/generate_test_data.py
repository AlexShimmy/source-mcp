
import os
import random
import json
from pathlib import Path

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def generate_python_file(path: Path, index: int):
    class_name = f"Service{index}"
    method_name = f"process_data_{index}"
    content = f"""
class {class_name}:
    \"\"\"
    Service class number {index} designed to handle specific data processing tasks.
    It is part of the core infrastructure for module {index % 5}.
    \"\"\"
    
    def {method_name}(self, data: dict) -> bool:
        \"\"\"
        Processes the input dictionary and returns True if successful.
        Key features:
        - Validates schema {index}
        - Transforms values using algorithm_v{index}
        \"\"\"
        # TODO: Implement validation logic for schema {index}
        if not data:
            return False
            
        print(f"Processing {{len(data)}} items using {class_name}")
        return True

def helper_{index}():
    \"\"\"
    Helper function number {index}.
    \"\"\"
    pass
"""
    path.write_text(content)

def generate_markdown_file(path: Path, index: int):
    topics = ["Deployment", "Architecture", "API Reference", "Contributing", "Security"]
    topic = topics[index % len(topics)]
    content = f"""# {topic} Guide {index}

## Introduction
This document covers the {topic} aspects of the system. It is critical for version {index}.0.

## Key Concepts
- **Concept {index}A**: The primary driver for {topic}.
- **Concept {index}B**: Secondary helper for {index}A.

## usage
To use this feature, run:
```bash
./run_feature_{index}.sh --mode=production
```

> [!NOTE]
> Ensure you have env var FEATURE_{index}_ENABLED set to true.
"""
    path.write_text(content)

def generate_text_file(path: Path, index: int):
    sentences = [
        f"The quick brown fox jumps over the lazy dog {index} times.",
        f"Error: Connection timeout at node {index}.",
        f"User {index} logged in successfully.",
        f"Metric {index} exceeded threshold.",
        f"Observation: The system behavior is stable at iteration {index}."
    ]
    # Inject a "needle" in some files
    if index % 10 == 0:
        sentences.append(f"NEEDLE_FOUND: special_secret_code_{index}")
        
    content = "\n".join(sentences)
    path.write_text(content)

def generate_json_file(path: Path, index: int):
    data = {
        "config_id": index,
        "enabled": (index % 2 == 0),
        "params": {
            "timeout": 100 + index,
            "retries": 3,
            "host": f"server-{index}.local"
        },
        "tags": [f"tag-{i}" for i in range(5)]
    }
    path.write_text(json.dumps(data, indent=2))

def main():
    root = Path(__file__).parent.parent.parent / "docs"
    # Ensure docs directory exists
    ensure_dir(root)
    
    # Clean up existing generated files if needed, or just overwrite/add
    # Let's generate 60 files to be safe (>50)
    print(f"Generating files in {root}...")
    
    generators = [
        (generate_python_file, "src/modules", ".py"),
        (generate_markdown_file, "guides", ".md"),
        (generate_text_file, "logs", ".txt"),
        (generate_json_file, "configs", ".json")
    ]
    
    for i in range(60):
        gen_func, subdir, ext = generators[i % len(generators)]
        
        # Create subdirectories for more realism
        target_dir = root / subdir
        ensure_dir(target_dir)
        
        filename = f"generated_file_{i}{ext}"
        filepath = target_dir / filename
        
        gen_func(filepath, i)
        
    print("Done generating 60 files.")

if __name__ == "__main__":
    main()
