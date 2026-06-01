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

### 3. Content-Based Filtering (Feature Vectorization & Profiles)
This method models items based on their textual features (genres, descriptions) and maps user profiles into the exact same vector space. It executes through a rigorous three-stage mathematical pipeline:

#### Step A: TF-IDF Item Vectorization (Space Modeling)
Extracts a vocabulary of $o$ terms from all item descriptions. Each item $i_l$ is transformed into a frequency-inverse document density vector inside a global matrix $M$ of dimensions $(m \times o)$:

$$\text{TF-IDF}(t_p, i_l) = \text{TF}(t_p, i_l) \times \log\left(\frac{1 + |I|}{1 + \text{DF}(t_p)}\right)$$

#### Step B: Dynamic User Preference Profiling ($Q_u$)
Instead of using static interaction logs, the system dynamically builds the user profile vector $Q_u$ (dimension $1 \times o$). It computes the dot product between the user's historical explicit ratings vector $R_u$ and the sub-matrix of items they have already interacted with ($M_{\text{rated}}$):

$$Q_u = R_u \cdot M_{\text{rated}}$$

#### Step C: Affinity Similarity Prediction ($S_u$) & Final Scoring
To forecast the user's affinity across all items simultaneously, the engine executes a matrix multiplication multiplying the entire corpus matrix $M$ by the transpose of the user profile vector ($Q_u^T$), resulting in the similarity matrix $S_u$ (dimension $m \times 1$):

$$S_u = M \cdot Q_u^T$$

Unfolding the matrix operation to calculate each individual item affinity $s_i$:

$$\begin{bmatrix} s_1 \\ s_2 \\ \vdots \\ s_m \end{bmatrix}_{(m \times 1)} = \begin{bmatrix} m_{1,1} & m_{1,2} & \cdots & m_{1,o} \\ m_{2,1} & m_{2,2} & \cdots & m_{2,o} \\ \vdots & \vdots & \ddots & \vdots \\ m_{m,1} & m_{m,2} & \cdots & m_{m,o} \end{bmatrix}_{(m \times o)} \cdot \begin{bmatrix} q_1 \\ q_2 \\ \vdots \\ q_o \end{bmatrix}_{(o \times 1)}$$

$$s_i = m_{i,1}q_1 + m_{i,2}q_2 + \dots + m_{i,o}q_o$$

## 🛠️ Installation & Architecture

1. Install the baseline data science dependencies listed in the requirements file:
   ```bash
   pip install -r requirements.txt