# zlearn

**zlearn** is a simple neural network library written in Rust, designed to offer both simplicity and high-level control to the user.

## Features

- **High user control**  
- **Matrix class included**  
- **Backpropagation and feedforward support**  

## XOR Example

```rust
use zlearn::activation::SIGMOID;
use zlearn::network::Network;

fn main() {
    // XOR input and target data
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];

    // Create a network: 2 input neurons, 2 hidden neurons, 1 output neuron
    let mut network = Network::new(vec![2, 2, 1], SIGMOID, 0.5);

    // Train the network for 10,000 epochs
    network.train(inputs.clone(), targets.clone(), 10_000);

    // Test the network
    println!("\nTesting XOR after training:");
    for i in 0..inputs.len() {
        let output = network.feed_forward(inputs[i].clone());
        println!(
            "Input: [{:.1}, {:.1}]  Output: {:.3}  Expected: [{:.1}]",
            inputs[i][0], inputs[i][1], output[0], targets[i][0]
        );
    }
}