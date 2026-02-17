# Process Data Script

A bash script to batch process JSON or HTML benchmark data files through the GuideLLM model trainer.

## Overview

The `process_data.sh` script automates the processing of multiple JSON or HTML benchmark data files by running the GuideLLM trainer ([demos/guidellm/main.go](../demos/guidellm/main.go) or [demos/guidellm-html/main.go](../demos/guidellm-html/main.go)) on each file individually and saving the output to text files with matching names.

## Usage

### Basic Syntax

```bash
./process_data.sh <input_dir> <output_dir>
```

### Parameters

| Parameter    | Required | Description                                     | Default     |
| ------------ | -------- | ----------------------------------------------- | ----------- |
| `input_dir`  | Yes      | Directory containing benchmark data files       | -           |
| `output_dir` | Yes      | Directory where output text files will be saved | -           |

### Example

```bash
./process_data.sh experiments/exp1/data experiments/exp1/results
```

This will:

- Read input files from `experiments/exp1/data/`
- Write output to `experiments/exp1/results/`
- Echo a summary of the results across all processed files (may be imported into a spreadsheet)
