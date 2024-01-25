use polars::export::rayon::prelude::*;
use polars::prelude;
use polars::prelude::*;
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use stopwords::{Language, Spark, Stopwords};
use text_processing::create_vocabulary;

use crate::text_processing::{clean_text, doc2bow};
mod text_processing;

fn bow() {
    let file_path = "./data/product_ideas.csv";

    let df = CsvReader::from_path(file_path)
        .expect("Failed to read CSV file")
        .infer_schema(None)
        .finish()
        .expect("Failed to parse CSV data");

    // Select the `coreSolution` column and take the first value
    let core_solution_series = df.column("coreSolution").expect("Column not found");

    // Convert to strongly typed ChunkedArray
    let core_solution_series_str = core_solution_series.str().expect("not str");

    let bow_column: Vec<_> = core_solution_series_str
        .par_iter() // Parallel iterator
        .map(|opt_val| opt_val.map(|val| doc2bow(val)).unwrap_or_default())
        .collect();
}

fn test() {
    // Load stopwords
    let stops: HashSet<_> = Spark::stopwords(Language::English)
        .unwrap()
        .iter()
        .cloned()
        .collect();

    // Example text
    let text = "Broccoli is good to eat. I have 5 broccoli heads.";

    // Clean text
    let cleaned_text = clean_text(text, &stops);

    println!("Original text: {}", text);
    println!("Cleaned text: {}", cleaned_text);
}

fn main() {
    let file_path = "./data/product_ideas.csv";

    // Load stopwords
    let stops: HashSet<_> = Spark::stopwords(Language::English)
        .unwrap()
        .iter()
        .cloned()
        .collect();

    // Load df
    let df = CsvReader::from_path(file_path)
        .expect("Failed to read CSV file")
        .infer_schema(None)
        .finish()
        .expect("Failed to parse CSV data");

    // Select the `coreSolution` column and take the first value
    let core_solution_series = df.column("coreSolution").expect("Column not found");

    // Convert to strongly typed ChunkedArray
    let core_solution_series_str = core_solution_series.str().expect("not str");

    // Clean each entry in the series and collect the results into a vector
    let cleaned_docs: Vec<String> = core_solution_series_str
        .par_iter()
        .map(|opt_val| {
            opt_val
                .map(|val| clean_text(val, &stops))
                .unwrap_or_default()
        })
        .collect();

    // Create a vocabulary of all unique words
    let vocab_list = create_vocabulary(&cleaned_docs);

    // Create a HashMap for word to index mapping
    let word_index: HashMap<_, _> = vocab_list
        .iter()
        .enumerate()
        .map(|(idx, word)| (word.clone(), idx))
        .collect();

    // Initialize a Document-Term  DataFrame with zeros
    let mut dtm: DataFrame = DataFrame::new(
        vocab_list
            .iter()
            .map(|word| UInt32Chunked::from_slice(word, &vec![0; cleaned_docs.len()]).into_series())
            .collect(),
    )
    .expect("msg");

    let num_documents = cleaned_docs.len();
    let num_words = vocab_list.len();
    let num_topics = 5; // The number of topics
    let num_iterations = 3;

    // Initialize parameters

    // 1. Document-Topic Distribution: This is a matrix where each row represents a document and each
    //  column represents a topic. The value in each cell is the probability of the topic in the corresponding
    //  document.
    let mut doc_topic_dist = vec![vec![0f64; num_topics]; num_documents];
    let mut rng = thread_rng();
    let topic_dist = Uniform::from(0..num_topics);
    for doc_dist in doc_topic_dist.iter_mut() {
        let mut topic_counts = vec![0; num_topics];
        for topic_count in topic_counts.iter_mut() {
            *topic_count = topic_dist.sample(&mut rng);
        }
        let total: usize = topic_counts.iter().sum();
        for (i, count) in topic_counts.iter().enumerate() {
            doc_dist[i] = *count as f64 / total as f64;
        }
    }

    // 2. Topic-Word Distribution: This is a matrix where each row represents a topic and each column a word.
    //  The value in each cell is the probability of the word in the corresponding topic.
    let mut topic_word_dist = vec![vec![0f64; num_words]; num_topics];
    let word_dist = Uniform::from(0..num_words);
    for topic_dist in topic_word_dist.iter_mut() {
        let mut word_counts = vec![0; num_words];
        for word_count in word_counts.iter_mut() {
            *word_count = word_dist.sample(&mut rng);
        }
        let total: usize = word_counts.iter().sum();
        for (i, count) in word_counts.iter().enumerate() {
            topic_dist[i] = *count as f64 / total as f64;
        }
    }

    // 3. Word-Topic Assignment: This is an auxilliary structure to track which topic is assigned to
    //  each word in document.

    // let mut word_topic_assignment: HashMap<(usize, usize), usize> = HashMap::new();
    // for (doc_idx, doc) in dtm.iter().enumerate() {
    //     for (word_idx, _) in doc.iter().enumerate() {
    //         let topic = topic_dist.sample(&mut rng);
    //         word_topic_assignment.insert((doc_idx, word_idx), topic);
    //     }
    // }

    let mut word_topic_assignment = vec![vec![0; num_words]; num_documents];
    for doc_idx in 0..num_documents {
        for word_idx in 0..num_words {
            let topic = topic_dist.sample(&mut rng);
            word_topic_assignment[doc_idx][word_idx] = topic;
        }
    }

    // LDirichlet prior parameters
    let alpha = 0.1; 
    let beta = 0.01;

    // LDA Algorithm
    let mut rng = thread_rng();

    for _ in 0..num_iterations {
        for doc_idx in 0..num_documents {
            for word_idx in 0..num_words {
                let current_topic = word_topic_assignment[doc_idx][word_idx];
            }
        }
    }
    

    println!("{:?}", word_topic_assignment);
}

// // Get first row
// let first_value = core_solution_series_str.get(0);

// Check if first_value is Some and then split into words
// let words_list = match first_value {
//     Some(value) => value.split_whitespace().collect::<Vec<&str>>(),
//     None => Vec::new(),
// };
