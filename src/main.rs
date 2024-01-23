use polars::export::rayon::prelude::*;
use polars::prelude;
use polars::prelude::*;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use stopwords::{Language, Spark, Stopwords};

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
    let core_solution_cleaned: Vec<String> = core_solution_series_str
        .par_iter()
        .map(|opt_val| {
            opt_val
                .map(|val| clean_text(val, &stops))
                .unwrap_or_default()
        })
        .collect();

    println!("{:?}", core_solution_cleaned);
}

// // Get first row
// let first_value = core_solution_series_str.get(0);

// Check if first_value is Some and then split into words
// let words_list = match first_value {
//     Some(value) => value.split_whitespace().collect::<Vec<&str>>(),
//     None => Vec::new(),
// };
