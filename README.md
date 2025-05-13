# Do Email Writing Styles Influence Response Time? <!-- omit in toc -->

**Author:** Gio Choi  

This repository accompanies the research project **“Do Email Writing Styles Influence Response Time?”** (working paper) and its earlier **proposal**. The work combines large‑scale organizational email data, large‑language‑model (LLM) text annotation, and hierarchical / network‑aware causal methods to estimate how an email’s linguistic **style**—in particular *casualness, directness,* and *emotional tone*—affects how quickly recipients reply.

---

## Table of contents <!-- omit in toc -->
1. [Project description](#project-description)
2. [Mock-up findings](#mock-up-findings)
3. [Repository layout](#repository-layout)
4. [Installation](#installation)
5. [How to cite](#how-to-cite)

---

## Project description

In this study, I investigate whether—and to what extent—the linguistic style of workplace emails drives the speed with which recipients reply. Building on two large, archival corpora—the Enron Email Corpus (over half a million cleaned one‑to‑one exchanges) and the Avocado Research Email Collection—I first preprocess each dataset to isolate discrete message–reply pairs and calculate response latencies in hours. I then leverage a locally hosted LLaMA‑3 2 × 3 B model (served via Ollama and orchestrated with LangChain) to assign binary style labels—casual versus non‑casual, direct versus indirect, and emotional versus neutral—to each email. To estimate causal effects, I fit a hierarchical linear model that nests individual emails within sender–recipient dyads, apply MRQAP permutation tests to account for network dependencies, and use inverse‑probability weighting to correct for selection bias toward quick or slow replies. By triangulating these methods, I aim to isolate the independent impact of tone and phrasing on response time, thereby offering novel, data‑driven guidance for writing more effective, timely emails in organizational settings.

---

## Mock-up Findings

| Style contrast | Estimated Δ hours<sup>†</sup> | 95 % CI |
| -------------- | ----------------------------- | ------- |
| **Casual vs. Non‑casual** | **−1.2 h** | [−2.0, −0.4] |
| **Direct vs. Non‑direct** | −0.7 h | [−1.3, −0.1] |
| Emotional vs. Non‑emotional | −0.4 h | [−0.9, +0.1] |

<sup>† Mocked point estimates from simulated HLM, holding covariates at medians.</sup>

![Unknown-8](https://github.com/user-attachments/assets/86a3aa11-a308-4e50-96c4-86906d27c5d4)


*Casual wording alone trims roughly **one hour** off expected reply time, even after controlling for urgency, hierarchy gaps, topics, and temporal rhythms.*

> **Key takeaway:** Language that *reduces social distance (casual)* or *clarifies intent (direct)* materially improves responsiveness. Emotional tone shows a weaker, more context‑dependent pattern.

Full simulation tables and robustness plots live in `figures/` and the `paper/` directory.

---

## Repository layout
.
├── data/                     
│   ├── raw/                  # (git‑ignored) original Enron & Avocado dumps  
│   └── processed/            # (git‑ignored) cleaned .parquet/.pkl datasets  
├── data_cleansing.ipynb      # Jupyter notebook for preprocessing & cleansing  
├── regression.py             # (git‑ignored) Causal Inference
├── labeling.py               # LLM‑based style labeling (Ollama + LangChain)  
├── position_extract.py       # Sender job‑title & seniority extraction script  
└── requirements.txt          # Python dependencies  

## Installation
```bash
python -m venv .your_venv
source .your_venv/bin/activate
pip install -r requirements.txt
```

## How to cite

@working{choi2025emailstyle,
  author  = {Choi, Jiwoong},
  title   = {Do Email Writing Styles Influence Response Time?},
  year    = {2025},
  note    = {GitHub repository, https://github.com/Gio-Choi/Perspectives_on_Computaional_Research_Gio}
}
