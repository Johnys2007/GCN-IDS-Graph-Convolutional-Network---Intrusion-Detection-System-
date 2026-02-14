# GGD: Graph-Based Guest Detection & Intrusion Analysis 🛡️🤖

A Proof of Concept (PoC) using **Graph Convolutional Networks (GCN)** to detect malicious nodes within a network infrastructure.

## 📌 Project Overview
This project demonstrates how Artificial Intelligence can be applied to **Network Security**. Instead of analyzing IP traffic in isolation, this model uses **Graph Neural Networks** to analyze the relationships and communication patterns between devices.

By viewing the network as a Graph (Nodes = Devices, Edges = Communications), the model can identify "infected neighborhoods" and detect lateral movement of threats.

## 🛠️ Tech Stack
- **Python**
- **PyTorch Geometric** (Deep Learning on Graphs)
- **NetworkX** (Graph Visualization)
- **Matplotlib** (Plotting results)

## 🚀 Logic & Math
The project uses the **GCN (Graph Convolutional Network)** architecture. The core idea is "Message Passing":
- Each node gathers features from its neighbors.
- The model learns to classify nodes as **Normal** or **Malicious** based on their connectivity patterns and features.
- This approach is highly effective for detecting Botnets and Advanced Persistent Threats (APTs).

## 💻 How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt