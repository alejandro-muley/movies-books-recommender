# Movies & Books Recommender Engine 🎬 📚

A highly modular recommendation system built in Python utilizing Object-Oriented Programming (OOP) principles and mathematical modeling. This engine is designed to handle multiple distinct datasets and execute personalized recommendation strategies based on user interactions and item characteristics.

## 🚀 Key Features

* **Algorithmic Diversity:** Implements three types of core engines: Simple (Bayesian correction), Collaborative Filtering (Cosinus distance similarity), and Content-Based Filtering (TF-IDF mapping).
* **Multi-Dataset Support:** Out-of-the-box compatibility with the **MovieLens 100k** dataset and the **Book Recommendation Dataset**.
* **Enterprise Practices:** Built with automated execution tracking via Python's `logging` library and binary state persistence through `pickle` serialization.

---

## 📊 Algorithmic Breakdown

## 📊 Core Recommender Engines

The system implements a robust mathematical ecosystem split across three main algorithmic approaches, with full multi-dimensional array operations fully documented and optimized within the source code:

### 1. Simple Recommendation Engine
* **Methodology:** Popularity-based scoring.
* **Key Implementation:** Implements a **Bayesian-Corrected Scoring Formula** to neutralize cold-start biases, ensuring items with low interaction rates do not skew top recommendations.

### 2. Collaborative Filtering Engine
* **Methodology:** User-to-User interaction profiles.
* **Key Implementation:** Computes directional affinity matrices using **Cosinus Distance Similarity** metrics applied over explicit interaction vectors.

### 3. Content-Based Filtering Engine
* **Methodology:** Text feature vectorization and user preference alignment.
* **Key Implementation:** Advanced data-pipeline executing high-speed matrix multiplications ($M \cdot Q_u^T$). It leverages **TF-IDF (Term Frequency - Inverse Document Frequency)** item representations alongside dynamic user profiling vectors to forecast precise affinity scores across the entire corpus.

## 🛠️ Installation & Architecture

1. Install the baseline data science dependencies listed in the requirements file:
   ```bash
   pip install -r requirements.txt