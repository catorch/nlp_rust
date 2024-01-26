use crate::lda::sample_discrete;
use crate::text_processing::{clean_text, doc2bow};
use ndarray::s;
use ndarray::{Array2, Axis};
use ndarray_rand::RandomExt;
use polars::export::rayon::prelude::*;
use polars::prelude;
use polars::prelude::*;
use rand::distributions::{Distribution, Uniform, WeightedIndex};
use rand::thread_rng;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::num;
use stopwords::{Language, Spark, Stopwords};
use text_processing::create_vocabulary;
mod lda;
mod text_processing;

fn bow() {
    let file_path = "./data/filtered_product_ideas.csv";

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

    // Generate a word count map
    let mut word_count_map: HashMap<String, usize> = HashMap::new();
    for doc in &cleaned_docs {
        let bow = doc2bow(doc);
        for (word, count) in bow {
            *word_count_map.entry(word).or_insert(0) += count;
        }
    }

    // Filter words that have at least 10 counts
    let filtered_vocab: Vec<String> = word_count_map
        .into_iter()
        .filter(|&(_, count)| count >= 20)
        .map(|(word, _)| word)
        .collect();

    // Create a vocabulary of all unique words
    let vocab_list = create_vocabulary(&cleaned_docs);

    // Create a HashMap for word to index mapping
    let word_index: HashMap<_, _> = filtered_vocab
        .iter()
        .enumerate()
        .map(|(idx, word)| (word.clone(), idx))
        .collect();

    // Initialize a Document-Term  DataFrame with zeros
    let mut dtm: DataFrame = DataFrame::new(
        filtered_vocab
            .iter()
            .map(|word| UInt32Chunked::from_slice(word, &vec![0; cleaned_docs.len()]).into_series())
            .collect(),
    )
    .expect("msg");

    let num_documents = cleaned_docs.len();
    let num_words = filtered_vocab.len();
    let num_topics = 3; // The number of topics
    let num_iterations = 3;
    let mut rng = thread_rng();

    println!("{:?}", num_words);

    // Initialize parameters

    // 1. Document-Topic Distribution: This is a matrix where each row represents a document and each
    //  column represents a topic. The value in each cell is the probability of the topic in the corresponding
    //  document.
    let mut doc_topic_dist: Array2<f64> = Array2::random_using(
        (num_documents, num_topics),
        Uniform::new(0.0, 1.0),
        &mut rng,
    );

    // Normalize each row to get probabilities
    for mut row in doc_topic_dist.axis_iter_mut(Axis(0)) {
        let sum: f64 = row.sum();
        row.mapv_inplace(|x| x / sum);
    }

    // 2. Topic-Word Distribution: This is a matrix where each row represents a topic and each column a word.
    //  The value in each cell is the probability of the word in the corresponding topic.
    let mut topic_word_dist: Array2<f64> =
        Array2::random_using((num_topics, num_words), Uniform::new(0.0, 1.0), &mut rng);

    // Normalize each row
    for mut row in topic_word_dist.axis_iter_mut(Axis(0)) {
        let sum: f64 = row.sum();
        row.mapv_inplace(|x| x / sum);
    }

    // 3. Word-Topic Assignment: This is an auxilliary structure to track which topic is assigned to
    //  each word in document.

    let topic_dist = Uniform::new(0, num_topics);
    let mut word_topic_assignment: Array2<usize> =
        Array2::random_using((num_documents, num_words), topic_dist, &mut rng);

    // LDirichlet prior parameters
    let alpha = 15.0;
    let beta = 0.01;

    let alpha_array = Array2::from_elem((num_documents, num_topics), alpha);
    let beta_array = Array2::from_elem((num_topics, num_words), beta);

    for _ in 0..num_iterations {
        for doc_idx in 0..num_documents {
            for word_idx in 0..num_words {
                let current_topic = word_topic_assignment[[doc_idx, word_idx]];

                // Decrement counts
                doc_topic_dist[[doc_idx, current_topic]] -= 1.0;
                topic_word_dist[[current_topic, word_idx]] -= 1.0;

                // Vectorized computation of probabilities
                let prob_topic_doc = (&doc_topic_dist.slice(s![doc_idx, ..])
                    + &alpha_array.slice(s![doc_idx, ..]))
                    / (&doc_topic_dist.slice(s![doc_idx, ..]).sum() + alpha * num_topics as f64);
                let prob_word_topic = (&topic_word_dist.slice(s![.., word_idx])
                    + &beta_array.slice(s![.., word_idx]))
                    / (&topic_word_dist.slice(s![.., word_idx]).sum() + beta * num_words as f64);
                let probabilities = &prob_topic_doc * &prob_word_topic;

                // Normalize probabilities
                let sum_probabilities: f64 = probabilities.iter().sum();

                // Handle the case where sum of probabilities is zero
                if sum_probabilities == 0.0 {
                    continue;
                }

                let normalized_probabilities: Vec<f64> = probabilities
                    .iter()
                    .map(|p| p / sum_probabilities)
                    .collect();

                // Handle invalid weights
                if normalized_probabilities
                    .iter()
                    .any(|&p| p.is_nan() || p <= 0.0)
                {
                    continue;
                }

                // Sample a new topic
                let dist = WeightedIndex::new(&normalized_probabilities);

                if let Ok(dist) = dist {
                    let new_topic = dist.sample(&mut rng);
                    word_topic_assignment[[doc_idx, word_idx]] = new_topic;

                    // Increment counts
                    doc_topic_dist[[doc_idx, new_topic]] += 1.0;
                    topic_word_dist[[new_topic, word_idx]] += 1.0;
                } else {
                    // Handle error in creating WeightedIndex

                    println!(
                        "Error creating WeightedIndex for doc_idx: {}, word_idx: {}",
                        doc_idx, word_idx
                    );
                }
            }
        }
    }

    // Extract top 20 words for each topic
    for topic_idx in 0..num_topics {
        let mut word_probabilities: Vec<(usize, f64)> = topic_word_dist
            .index_axis(Axis(0), topic_idx)
            .iter()
            .enumerate()
            .map(|(word_idx, &prob)| (word_idx, prob))
            .collect();

        word_probabilities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_words: Vec<String> = word_probabilities
            .iter()
            .take(20)
            .map(|&(word_idx, _)| filtered_vocab[word_idx].clone())
            .collect();

        println!("Topic {}: {:?}", topic_idx + 1, top_words);
    }
}

// // Get first row
// let first_value = core_solution_series_str.get(0);

// Check if first_value is Some and then split into words
// let words_list = match first_value {
//     Some(value) => value.split_whitespace().collect::<Vec<&str>>(),
//     None => Vec::new(),
// };
