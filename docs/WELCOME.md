# Welcome to 3-Adic ML! 🌌

Welcome to the **3-Adic ML** project. We are exploring a fascinating intersection of pure mathematics, geometry, and deep learning. If you're new here, this guide will help you understand what we're doing and why it matters.

## 🧠 The Big Idea: "Meaning = Geometry"

Most AI models learn patterns from data, but they often struggle with **hierarchy**. For example, a "Golden Retriever" is a type of "Dog," which is a "Mammal," which is an "Animal." 

In this project, we believe that **hierarchy should be built into the shape of the space where the AI "thinks."**

### 1. P-Adic Numbers (The Logic)
We use **3-adic valuation** as our mathematical foundation. It's a way of measuring "divisibility." In our system, some numbers are "more fundamental" than others based on how many times they can be divided by 3. This gives us a natural, mathematical tree structure.

### 2. Hyperbolic Geometry (The Map)
Imagine a map where the further you go from the center, the more "room" there is. That's a **Poincaré Ball** (a model of hyperbolic geometry). It is the perfect home for tree structures because it can fit an infinite number of branches without them getting squished.

### 3. The Result
We train an AI (a Variational Autoencoder or VAE) to map data into this hyperbolic space. 
- **Fundamental things** (high valuation) are placed near the **center**.
- **Specific details** (low valuation) are placed near the **edges**.
- **Similar things** are placed in the same **direction**.

Hierarchy isn't just learned; it's **emergent from the geometry**.

## 🚀 Why are we doing this?

*   **Better AI Reasoning:** Models that understand hierarchy natively can reason better about complex structures like code, biology, or language.
*   **Scientific Discovery:** We're applying this to bioinformatics to understand how proteins and genes are organized.
*   **Efficiency:** We can represent complex relationships in very few dimensions.

## 🛠️ How to Explore

1.  **Read the [README](../README.md):** For the technical setup and architecture.
2.  **Check the Visualizations:** During training, we generate interactive 3D maps (HTML files) showing how the AI organizes the data.
3.  **Explore the [FAQ](FAQ.md):** For answers to common questions.
4.  **Join the Research:** Look at our [STATUS.md](STATUS.md) to see what we're currently working on (Phase 10+).

## 🌍 Community

We are a small, focused research group. We value technical integrity and mathematical beauty. Whether you're a math enthusiast, a deep learning engineer, or just curious, we're glad you're here!

## 🔢 What are 3-Adic Numbers?

To understand our project, you don't need to be a math expert—just imagine numbers in a different way.

Normally, we think two numbers are "close" if their difference is small (like 1 and 2). In the **p-adic** world (where $p=3$ in our case), closeness is measured by **divisibility**.
- The number **3** is more "fundamental" or "smaller" than 1.
- The number **9** ($3^2$) is even more fundamental.
- The number **27** ($3^3$) is more so still.

This creates a **valuation**. The more times a number is divisible by 3, the closer it is to the "center" of our system. Mathematically, this generates an **ultrametric space**: a structure that behaves like an infinite tree where every branch splits into three.

## 🧬 The Bio Connection: Nucleotides and Codons

Why use base-3 math for life? Biology is hierarchical by nature:

1.  **Codons and Triplets**: Life is written in clusters of three. A **codon** is a sequence of 3 nucleotides that encodes an aminoacid. This "triplet" structure resonates perfectly with the ternary logic (base 3) of our system.
2.  **The Hierarchy of Evolution**: DNA is not just a string of letters; it is a historical record. Mutations that occur in critical positions are more "fundamental" than others. 3-adic valuation allows us to map these priorities:
    *   **Level 0**: Surface-level changes.
    *   **Higher Levels**: Structural changes in proteins or vital functions.
3.  **Genome Geometry**: By mapping nucleotides (A, C, G, T) to 3-adic coordinates, we can visualize the genome not as flat text, but as a **geometric landscape** in a Poincaré Ball.

## 🌡️ How can this change Bioinformatics?

The application of 3-adic VAEs to bioinformatics (our sister project **[3-Adic Bioinformatics](https://github.com/gesttaltt/3-adic-bioinformatics)**) aims to revolutionize the field in three ways:

*   **Semantic Compression**: Instead of storing gigabytes of "dead" sequences, we store the **geometry of meaning**. We can compress entire genomes while preserving the hierarchical relationships between genes.
*   **Protein Folding Prediction**: Proteins fold following energy hierarchies. Hyperbolic geometry is ideal for modeling how small substructures combine into large structures without "colliding" in the computational space.
*   **Ultra-fast Search**: In an ultrametric space, finding similar sequences is orders of magnitude faster. We don't compare letter by letter; we compare "branches of the tree" in hyperbolic space.

---

*"Life does not read text; life inhabits a geometry. We are here to decipher it."*
