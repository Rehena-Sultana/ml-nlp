from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Annotated, Literal
import re
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data (only runs once, then cached)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

app = FastAPI(title="Text Preprocessing API")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


class TextInput(BaseModel):
    text: Annotated[str, Field(..., min_length=1, description="Text to preprocess")]


class TechniqueInput(BaseModel):
    text: Annotated[str, Field(..., min_length=1, description="Text to preprocess")]
    technique: Annotated[
        Literal[
            "lowercase",
            "remove_punctuation",
            "remove_numbers",
            "remove_extra_spaces",
            "remove_html_tags",
            "remove_urls",
            "tokenize",
            "remove_stopwords",
            "stem",
            "lemmatize",
            "pos_tag",
        ],
        Field(..., description="Which single technique to apply"),
    ]


# ---------- individual cleaning functions ----------

def to_lowercase(text: str) -> str:
    return text.lower()


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_numbers(text: str) -> str:
    return re.sub(r'\d+', '', text)


def remove_extra_spaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def remove_html_tags(text: str) -> str:
    return re.sub(r'<.*?>', '', text)


def remove_urls(text: str) -> str:
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def full_clean(text: str) -> str:
    """Used internally before tokenize/stem/lemmatize/etc for the full pipeline."""
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_extra_spaces(text)
    return text


@app.get("/")
def home():
    return {"message": "Text Preprocessing API is running"}


@app.post("/preprocess")
def preprocess_text(data: TextInput):
    """Runs the FULL pipeline, step by step."""
    original_text = data.text

    no_html = remove_html_tags(original_text)
    no_urls = remove_urls(no_html)
    lowercased = to_lowercase(no_urls)
    no_punctuation = remove_punctuation(lowercased)
    no_numbers = remove_numbers(no_punctuation)
    cleaned_text = remove_extra_spaces(no_numbers)

    tokens = word_tokenize(cleaned_text)
    tokens_no_stopwords = [word for word in tokens if word.lower() not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in tokens_no_stopwords]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens_no_stopwords]
    pos_tags = nltk.pos_tag(tokens)

    result = {
        "original_text": original_text,
        "no_html": no_html,
        "no_urls": no_urls,
        "lowercased": lowercased,
        "no_punctuation": no_punctuation,
        "no_numbers": no_numbers,
        "cleaned_text": cleaned_text,
        "tokens": tokens,
        "tokens_without_stopwords": tokens_no_stopwords,
        "stemmed_words": stemmed_words,
        "lemmatized_words": lemmatized_words,
        "pos_tags": pos_tags,
    }

    return JSONResponse(status_code=200, content={"response": result})


@app.post("/preprocess/single")
def preprocess_single_technique(data: TechniqueInput):
    """Runs only ONE selected technique and returns just that output."""
    text = data.text
    technique = data.technique

    if technique == "lowercase":
        output = to_lowercase(text)

    elif technique == "remove_punctuation":
        output = remove_punctuation(text)

    elif technique == "remove_numbers":
        output = remove_numbers(text)

    elif technique == "remove_extra_spaces":
        output = remove_extra_spaces(text)

    elif technique == "remove_html_tags":
        output = remove_html_tags(text)

    elif technique == "remove_urls":
        output = remove_urls(text)

    elif technique == "tokenize":
        cleaned = full_clean(text)
        output = word_tokenize(cleaned)

    elif technique == "remove_stopwords":
        cleaned = full_clean(text)
        tokens = word_tokenize(cleaned)
        output = [word for word in tokens if word.lower() not in stop_words]

    elif technique == "stem":
        cleaned = full_clean(text)
        tokens = word_tokenize(cleaned)
        tokens_no_stopwords = [w for w in tokens if w.lower() not in stop_words]
        output = [stemmer.stem(word) for word in tokens_no_stopwords]

    elif technique == "lemmatize":
        cleaned = full_clean(text)
        tokens = word_tokenize(cleaned)
        tokens_no_stopwords = [w for w in tokens if w.lower() not in stop_words]
        output = [lemmatizer.lemmatize(word) for word in tokens_no_stopwords]

    elif technique == "pos_tag":
        cleaned = full_clean(text)
        tokens = word_tokenize(cleaned)
        output = nltk.pos_tag(tokens)

    return JSONResponse(
        status_code=200,
        content={"response": {"technique": technique, "output": output}},
    )