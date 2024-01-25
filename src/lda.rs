use rand::distributions::{Distribution, Uniform};

// Helper function to sample from a discrete distribution
pub fn sample_discrete(probabilities: &[f64], rng: &mut impl rand::Rng) -> usize {
    let dist = Uniform::new(0.0, 1.0);
    let mut cumulative = 0.0;
    let sample = dist.sample(rng);
    for (i, &prob) in probabilities.iter().enumerate() {
        cumulative += prob;
        if sample < cumulative {
            return i;
        }
    }
    probabilities.len() - 1
}