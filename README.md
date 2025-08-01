# Contract Review AI System

An intelligent contract review system powered by AI agents that analyzes legal documents using the United Educators checklist methodology. The system provides comprehensive clause analysis, risk assessment, and compliance evaluation.

## Features

- **Multi-Agent Analysis**: Three specialized AI agents work collaboratively
  - Contract Analysis Agent: Extracts and categorizes clauses
  - Risk Assessment Agent: Evaluates risks and assigns priority levels
  - United Educators Review Agent: Follows established checklist methodology

- **Comprehensive Review**: 
  - Clause-by-clause analysis with risk scoring
  - Compliance status assessment
  - Redline suggestions and recommendations
  - User attention scoring for prioritization

- **Knowledge-Based**: Uses United Educators contract review guidelines as reference

- **Structured Output**: Returns standardized analysis with categorized findings

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Windows/Linux/Mac environment

## Installation

### 1. Clone or download the project
```bash
git clone 
cd contract-reviewer
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies from **requirements.txt**
```bash
pip install -r requirements.txt
```

> A *requirements.txt* file is already provided in the repo so you don’t have to build it yourself. If you ever update packages and need to regenerate it, run:  
> `pip freeze > requirements.txt`

The script will also create `tmp/lancedb/` (vector DB) and `tmp/contract_memory.db` (SQLite storage) on first run.

## Usage
```bash
python "ia_reviewer_contract.py"
```
The console prints the structured review (Clause analyses, Risk summary, Recommendations, Questions for user).

## Customization
- **User context**: edit the `UserContext` object inside the script for personalized insights.  
- **Model choice**: change the `id` in each `OpenAIChat` instantiation if you wish to use another GPT model.

## Security Tips
- Never commit real API keys.  

You’re now ready to run a complete AI-driven contract review.
