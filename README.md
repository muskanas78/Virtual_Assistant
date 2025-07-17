# Virtual AI Assistant

A modular, locally-run Smart Virtual Assistant built with **Streamlit** and powered by **Ollama** LLMs. This assistant supports a wide range of language tasks including question answering, summarization, translation, JSON formatting, structured output generation, chain-of-thought reasoning, few-shot classification, and role-play interactions. The application uses prompt engineering and Jinja2 templating to ensure accuracy, control, and high-quality responses across all tasks.

## Problem Statement

Most language-based assistants lack structured prompt control, transparency, and extensibility. They either depend on cloud APIs, are expensive to scale, or fail to maintain accuracy across diverse NLP tasks. Furthermore, local deployment options for multi-functional virtual assistants are often fragmented or overly complex.

## Solution

This project presents a **streamlined, local-first virtual assistant** powered by Ollama models and guided by modular prompt templates. By integrating Streamlit for UI, Jinja2 for templating, and a clean modular backend, this assistant provides a lightweight yet powerful solution for performing and experimenting with structured NLP tasks, all while running locally.

## Features

### Modular Prompt Engine

* Template-based prompt construction using `jinja2`
* Clean separation of system and user instructions
* Designed for reusability, clarity, and accuracy

### Task Capabilities

* **Q\&A**: Responds to factual, technical, and general knowledge questions with short, accurate answers.
* **Summarization**: Condenses complex content into concise, meaningful summaries.
* **Translation**: Provides accurate translations between supported languages, preserving tone and intent.
* **Role-Play**: Simulates domain experts (e.g., Doctor, Engineer, Lawyer) to respond in context-specific tone.
* **JSON Formatter**: Converts plain text into valid JSON if structure is clear and unambiguous.
* **Few-Shot Classification**: Performs sentiment or intent classification using zero-shot or few-shot prompting.
* **Chain-of-Thought Reasoning**: Offers structured, step-by-step reasoning for problems and questions.
* **JSON Validator**: Validates user JSON against a predefined schema with detailed feedback.
* **Role-Play Comparison**: Compares responses between role-based and standard Q\&A for analysis.

## Installation

### Requirements

* Python 3.8+
* [Ollama](https://ollama.com/) installed and running locally (with models like `gemma3`, `qwen3:0.6b`, `mistral`, etc.)
* Streamlit
* Jinja2
* Requests

### Install Dependencies

```bash
pip install streamlit
```

### Run Ollama Server

Make sure Ollama is running and your desired model (e.g., `gemma3:latest`) is pulled:

```bash
ollama run gemma3
```

### Run the App

```bash
streamlit run virtual_assistant.py
```

## File Structure

```
.
├── virtual_assistant.py    # Main application entry
├── ai_bot.png              # Assistant icon
├── README.md
```

## Configuration

You can change the model and server configuration in the following section of the code:

```python
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:latest"
```

## Acknowledgments

* [Ollama](https://ollama.com/) for local LLM serving
* [Streamlit](https://streamlit.io/) for building a fast and interactive UI
* [Jinja2](https://jinja.palletsprojects.com/) for flexible prompt templating
