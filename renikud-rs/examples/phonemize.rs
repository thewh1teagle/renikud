/*
Reads Hebrew sentences from stdin (one per line), writes IPA to stdout (one per line).

Usage:
    echo "שלום עולם" | cargo run --release --example phonemize -- model.onnx
*/

use std::io::{self, BufRead, Write};
use renikud_rs::G2P;

fn main() -> anyhow::Result<()> {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| "model.onnx".to_string());
    let mut g2p = G2P::new(&model_path)?;
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    for line in stdin.lock().lines() {
        let line = line?;
        let ipa = g2p.phonemize(&line)?;
        writeln!(out, "{}", ipa)?;
    }
    Ok(())
}
