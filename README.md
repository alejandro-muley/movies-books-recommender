# Movies & Books Recommender Engine 🎬 📚

A highly modular recommendation system built in Python utilizing Object-Oriented Programming (OOP) principles and mathematical modeling. This engine is designed to handle multiple distinct datasets and execute personalized recommendation strategies based on user interactions and item characteristics.

## 🚀 Key Features

* **Algorithmic Diversity:** Implements three types of core engines: Simple (Bayesian correction), Collaborative Filtering (Cosinus distance similarity), and Content-Based Filtering (TF-IDF mapping).
* **Multi-Dataset Support:** Out-of-the-box compatibility with the **MovieLens 100k** dataset and the **Book Recommendation Dataset**.
* **Enterprise Practices:** Built with automated execution tracking via Python's `logging` library and binary state persistence through `pickle` serialization.

---

## 📊 Algorithmic Breakdown

### 1. Simple Recommendation (Popularity-Based)
To prevent cold-start biases where items with very few votes skew the top charts, the system implements a **Bayesian-corrected scoring formula**:
$$score = \left(\frac{\text{num\_vots}}{\text{num\_vots} + \text{min\_vots}} \cdot \text{avg\_item}\right) + \left(\frac{\text{min\_vots}}{\text{num\_vots} + \text{min\_vots}} \cdot \text{avg\_global}\right)$$

### 2. Collaborative Filtering (User-User Interaction)
Measures the directional affinity between users using the **Cosinus Distance** metric on matching item interaction vectors:
$$s(u,v) = \frac{\sum_{i \in I} p_{ui} \cdot p_{vi}}{\sqrt{\sum_{i \in I} p_{ui}^2} \sqrt{\sum_{i \in I} p_{vi}^2}}$$

### 3. Content-Based Filtering (Feature Vectorization)
Builds dynamic user preference profiles by computing the mathematical product of the item features matrix generated via **TF-IDF (Term Frequency - Inverse Document Frequency)** vectorization:
$$S_u = M Q_u^T$$

---

## 🛠️ Installation & Architecture

1. Install the baseline data science dependencies listed in the requirements file:
   ```bash
   pip install -r requirements.txt