# 📊 Sales Data Analysis & Customer Segmentation (August 2019)
This project performs in-depth analysis of August 2019 sales data using Python. It includes data cleaning, visualizations, and customer segmentation using KMeans clustering based on purchasing behavior.

---
## 🧾 Features

- ✅ Clean and preprocess sales data (handle nulls, types, duplicates)
- 📈 Visualize daily trends, top-selling products, and geographic patterns
- 🏙️ Sales analysis by city and day of the week
- 🔥 Correlation analysis between product price and quantity ordered
- 📦 KMeans clustering for customer segmentation
- 📐 Evaluate clusters using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score
- 📊 Interactive visualizations with Seaborn and Matplotlib

---

## 📁 Dataset

**Required file:** `Sales_August_2019.csv`

> Ensure this file is in the same directory as the Python script.

---

## ⚙️ Requirements

Install the required Python libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

---

## 🚀 How to Run

1. Clone or download this repository.
2. Place `Sales_August_2019.csv` in the project directory.
3. Run the script:
```bash
python your_script_name.py
```

---

## 📈 Visualizations

- 📆 Daily sales line plot
- 🛍️ Top 10 best-selling products (bar chart)
- 📅 Weekly sales and purchase pie charts
- 🏙️ City-wise sales comparison
- 🌡️ Correlation heatmap: Quantity Ordered vs Price
- 📊 Elbow Method to determine optimal clusters
- 👥 Cluster segmentation and insights

---

## 🧠 Customer Segments (KMeans Clustering)

| Cluster ID | Segment Description             |
|------------|----------------------------------|
| 0          | Moderate Spenders                |
| 1          | Frequent Low-Value Buyers        |
| 2          | Bulk Bargain Shoppers            |
| 3          | Luxury/Premium Buyers            |
| 4          | Occasional High-Spend Buyers     |
| 5          | Infrequent Low-Spenders          |

Each segment includes:
- Total money spent
- Total quantity ordered
- Number of orders
- Average selling price
- Revenue per order

---

## 📊 Cluster Evaluation Metrics

- **Silhouette Score** – measures clustering quality (higher is better)
- **Davies-Bouldin Index** – measures cluster separation (lower is better)
- **Calinski-Harabasz Score** – measures intra-cluster tightness and inter-cluster separation (higher is better)

---

