use regex::Regex;
use std::collections::HashMap;
use std::collections::HashSet;
use stopwords::{Language, Spark, Stopwords};

pub fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| word.to_lowercase())
        .collect()
}

pub fn doc2bow(document: &str) -> HashMap<String, usize> {
    let tokens = tokenize(document);
    let mut counts = HashMap::new();

    for token in tokens {
        *counts.entry(token).or_insert(0) += 1;
    }

    counts
}

pub fn clean_text(text: &str, stops: &HashSet<&str>) -> String {
    let punctuation_re = Regex::new(r"[^\w\s]").unwrap();
    let numbers_re = Regex::new(r"\d").unwrap();

    text.to_lowercase()
        .split_whitespace()
        .map(|s| punctuation_re.replace_all(s, "").to_string()) // Convert to owned String
        .map(|s| numbers_re.replace_all(&s, "").to_string())   // Convert to owned String
        .filter(|s| !stops.contains(s.as_str())) // Use as_str() for comparison
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn create_vocabulary(docs: &Vec<String>) -> Vec<String> {
    let mut vocabulary = HashSet::new();
    for doc in docs {
        for word in doc.split_whitespace() {
            vocabulary.insert(word.to_string());
        }
    }

    return vocabulary.into_iter().collect()
}