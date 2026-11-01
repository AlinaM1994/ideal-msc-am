# AI Hype or Sustainable Innovation? A Sentiment Analysis framework to investigate Media Coverage vs. Funding Outcomes in AI Startups

## Project Overview
the project aims to discover if there is a relationship between the usage of AI terminology in the startup communication and their investment success. 

## Project information
- Student: Alina Mazur
- Supervisor: Professor Alessandro Provetti
- Programme: MSc Data Science
- Institution: School of Computing and Mathematical Sciences, Birkbeck, University of London




1 November 2026

# Innovate UK RAG Pipeline: rag_pipeline.py

## Project Overview

This project explores Retrieval-Augmented Generation (RAG) for answering natural-language questions about publicly funded Innovate UK projects.

The current prototype uses a dataset of unique projects with publicly available descriptions.

## How the Pipeline Works

1. **Data preparation:** Project records are extracted from the Innovate UK Excel dataset, and duplicate project entries are removed.
2. **Embeddings:** The `all-MiniLM-L6-v2` model converts project descriptions into numerical vectors representing their meaning.
3. **Retrieval:** When a user asks a question, the system compares its embedding with the saved project embeddings and retrieves the five most semantically similar projects.
4. **Prompt augmentation:** The retrieved project titles, numbers and descriptions are added to the user's question as context.
5. **Answer generation:** `gpt-4o-mini` generates an answer using the retrieved information, with instructions to cite projects and avoid unsupported claims.

## Current Progress

The prototype works end-to-end and has been tested with questions about AI in public services and unsupported claims about healthcare outcomes.

## Current Limitations and Next Steps

The system currently retrieves only five projects and passes the first 800 characters of each description to the language model. This may exclude useful evidence.

The next stage is to examine how we can involve other datasets into our pipeline.
