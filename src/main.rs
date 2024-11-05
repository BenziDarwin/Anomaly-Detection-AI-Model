use csv::ReaderBuilder;
use ndarray::{Array1, Array2, s};
use linfa::prelude::*;
use linfa_trees::DecisionTree;
use std::collections::HashMap;
use std::error::Error;

// Function to map labels to numeric values for classification
fn encode_labels(labels: Vec<String>) -> (Array1<usize>, HashMap<String, usize>) {
    let mut label_map = HashMap::new();
    let mut current_label = 0;
    let encoded_labels: Vec<usize> = labels
        .into_iter()
        .map(|label| {
            *label_map.entry(label).or_insert_with(|| {
                let val = current_label;
                current_label += 1;
                val
            })
        })
        .collect();
    let array = Array1::from_vec(encoded_labels);
    (array, label_map)
}

// Function to read CSV into an ndarray with chunked reading
fn read_csv_to_ndarray(file_path: &str) -> Result<(Array2<f64>, Vec<String>), Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(file_path)?;
    
    // Pre-allocate vectors with capacity
    let mut features = Vec::new();
    let mut labels = Vec::new();
    let mut row_count = 0;
    
    // First pass to count rows and columns
    for result in reader.records() {
        let record = result?;
        row_count += 1;
        if row_count == 1 {
            // Pre-allocate based on first row
            let col_count = record.len() - 1;
            features = Vec::with_capacity(row_count * col_count);
            labels = Vec::with_capacity(row_count);
        }
        
        let row: Vec<f64> = record
            .iter()
            .take(record.len() - 1)
            .filter_map(|field| field.parse::<f64>().ok())
            .collect();
            
        if row.len() == record.len() - 1 {
            features.extend(row);
            labels.push(record[record.len() - 1].to_string());
        }
    }

    let num_cols = if !features.is_empty() { features.len() / labels.len() } else { 0 };
    let feature_array = Array2::from_shape_vec((labels.len(), num_cols), features)?;
    
    Ok((feature_array, labels))
}

// Function to split data into training and testing sets
fn split_data(data: Array2<f64>, targets: Array1<usize>) -> (Array2<f64>, Array1<usize>, Array2<f64>, Array1<usize>) {
    let num_samples = data.nrows();
    let split_at = (num_samples as f64 * 0.8) as usize;
    
    let x_train = data.slice(s![..split_at, ..]).to_owned();
    let y_train = targets.slice(s![..split_at]).to_owned();
    let x_test = data.slice(s![split_at.., ..]).to_owned();
    let y_test = targets.slice(s![split_at..]).to_owned();
    
    (x_train, y_train, x_test, y_test)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Increase stack size for the main thread
    #[cfg(target_os = "windows")]
    {
        use std::thread;
        let builder = thread::Builder::new()
            .stack_size(32 * 1024 * 1024); // 32MB stack
        builder.spawn(|| {
            if let Err(e) = run_main() {
                eprintln!("Error: {}", e);
            }
        })?.join().unwrap();
    }

    #[cfg(not(target_os = "windows"))]
    {
        run_main()?;
    }

    Ok(())
}

fn run_main() -> Result<(), Box<dyn Error>> {
    let file_path = "data/train.csv";
    
    println!("Reading data from CSV...");
    let (data, labels) = read_csv_to_ndarray(file_path)?;
    
    println!("Encoding labels...");
    let (encoded_labels, label_map) = encode_labels(labels);
    
    println!("Splitting data...");
    let (x_train, y_train, x_test, y_test) = split_data(data, encoded_labels);
    
    println!("Creating datasets...");
    let train_dataset = Dataset::from((x_train, y_train));
    let test_dataset = Dataset::from((x_test, y_test));
    
    println!("Training model...");
    let model = DecisionTree::params()
        .max_depth(Some(10)) // Limit tree depth to prevent stack overflow
        .min_weight_leaf(1.0)
        .fit(&train_dataset)
        .expect("Failed to train model");
    
    println!("Making predictions...");
    let predictions = model.predict(&test_dataset);
    
    let accuracy = predictions
        .iter()
        .zip(test_dataset.targets().iter())
        .filter(|(&pred, &actual)| pred == actual)
        .count() as f64
        / test_dataset.targets().len() as f64;
    
    println!("Model accuracy: {:.2}%", accuracy * 100.0);
    println!("Label encoding map: {:?}", label_map);
    
    Ok(())
}