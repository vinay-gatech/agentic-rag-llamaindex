
# Agentic RAG with LlamaIndex

This project demonstrates the integration of Retrieval-Augmented Generation (RAG) using the LlamaIndex framework. It leverages the OpenAI API and processes multiple research papers to answer queries and summarize information effectively.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Contributing](#contributing)
- [License](#license)

## Features

- Query specific research papers using natural language.
- Summarize documents and extract relevant information.
- Support for multiple documents through the Object Index.
- Utilizes OpenAI's GPT-3.5 Turbo for generating responses.

## Installation

To set up the project, ensure you have Python 3.7 or higher installed. Then, clone the repository and install the required packages:

```bash
git clone https://github.com/vinay-gatech/agentic-rag-llamaindex.git
cd agentic-rag-llamaindex
pip install -r requirements.txt
```

Make sure to set up your environment variables as needed, particularly for accessing the OpenAI API. You can create a `.env` file in the root directory:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Running the Project

You can interact with the project by executing the provided scripts. Here are some examples of how to run them:

1. **Single Document Agent**:
   ```bash
   python router_engine.py
   ```

2. **Multi-Document Agent**:
   ```bash
   python mulit-document-agent.py
   ```

3. **Agent Reasoning Loop**:
   ```bash
   python agent_reasoning_loop.py
   ```

### Querying Papers

You can customize the queries within the scripts to fetch specific information from the loaded research papers. For example, modify the `agent.query` method in the `agent_reasoning_loop.py` file to ask about different aspects of the papers.

## File Descriptions

- `utils.py`: Contains utility functions for loading documents and creating query tools.
- `router_engine.py`: Main script for querying a single document using the LlamaIndex framework.
- `mulit-document-agent.py`: Supports querying across multiple documents by creating an object index.
- `agent_reasoning_loop.py`: Implements a reasoning loop to interact with the loaded documents.
- `tool_calling.py`: Demonstrates how to use function tools for querying specific information from documents.

## Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, feel free to create an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
