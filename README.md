# CodeAlpha AI Chatbot

A simple **Java-based AI Chatbot** that uses **TF-IDF (Term Frequency–Inverse Document Frequency)** and **Cosine Similarity** to understand user intent and generate appropriate responses. This project was built as part of a **CodeAlpha internship task** to demonstrate basic Natural Language Processing concepts without using external AI APIs.

## Project Overview

The chatbot works by:

* Reading predefined intents and responses from a text file
* Converting user input and training phrases into TF-IDF vectors
* Calculating cosine similarity to find the closest matching intent
* Returning a response based on the best match

The chatbot runs entirely in the **console** and is implemented using **core Java only**.

## Features

* Intent-based chatbot
* TF-IDF vectorization from scratch
* Cosine similarity for intent matching
* Stopword removal and text preprocessing
* Threshold-based fallback response
* No external libraries or APIs required

## Project Structure

```
CodeAlpha_AI-Chatbot
│
├── src/
│   ├── Main.java              # Entry point of the application
│   └── ChatbotTFIDF.java      # Core chatbot logic (TF-IDF + cosine similarity)
│
├── data/
│   └── intents.txt            # Training data (intents and responses)
│
├── out/                       # Compiled class files
└── README.md
```

## How It Works

1. The chatbot loads `intents.txt`, which contains intent patterns and responses.
2. Each intent pattern is converted into a TF-IDF vector.
3. User input is processed in real time and converted into a TF-IDF vector.
4. Cosine similarity is calculated between user input and stored intents.
5. The intent with the highest similarity above a fixed threshold is selected.
6. If no intent crosses the threshold, a fallback response is shown.

## Requirements

* Java JDK 8 or higher
* Any IDE (IntelliJ IDEA, Eclipse, VS Code) or terminal

## How to Run

### Using Terminal

1. Clone the repository:

```
git clone https://github.com/shamky/CodeAlpha_AI-Chatbot.git
cd CodeAlpha_AI-Chatbot
```

2. Compile the project:

```
javac src/*.java
```

3. Run the chatbot:

```
java src.Main
```

### Using an IDE

1. Open the project in your IDE
2. Ensure `src` is marked as the source directory
3. Run `Main.java`

## Sample Interaction

```
You: hello
Bot: Hi! How can I help you today?

You: what is tf idf
Bot: TF-IDF stands for Term Frequency Inverse Document Frequency...
```

## Customization

* Add or modify intents in `data/intents.txt`
* Adjust similarity threshold in `ChatbotTFIDF.java`
* Extend stopword list for better preprocessing

## Limitations

* Rule-based intent matching
* No learning from conversations
* Limited vocabulary based on training data

## Future Improvements

* GUI or web-based interface
* Support for JSON-based intent files
* Dynamic learning
* Integration with advanced NLP libraries

## Author

**Shamky**

GitHub: [https://github.com/shamky](https://github.com/shamky)
