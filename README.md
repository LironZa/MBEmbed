# MBEmbed
Model for metagenomic gut microbiome representations

**"Self-Supervised Representation Learning for Microbiome Improves Downstream Prediction in Data-Limited and Cross-Cohort Settings"**
*Accepted to the ICML 2025 Workshop on Generative AI in Biology and the Workshop on Multi-modal Foundation Models and Large Language Models for Life Sciences*

**Read the paper here:** [https://openreview.net/forum?id=2JsJqQbjJn](https://openreview.net/forum?id=2JsJqQbjJn)

---

## Abstract

The gut microbiome plays a crucial role in human health, but machine learning applications in this field face significant challenges, including limited data availability, high dimensionality, and batch effects across different cohorts. While foundation models have transformed other biological domains, metagenomic data remains relatively under-explored despite its complexity and clinical importance. We developed self-supervised representation learning methods for gut microbiome metagenomic data by implementing multiple approaches on 85,364 samples, including masked autoencoders and novel cross-domain adaptation of single-cell RNA sequencing models. Systematic benchmarking against the standard practice in microbiome machine learning demonstrated significant advantages of our learned representations in limited-data scenarios, improving prediction for age (r = 0.14 vs. 0.06), BMI (r = 0.16 vs. 0.11), visceral fat mass (r = 0.25 vs. 0.18), and drug usage (PR-AUC = 0.81 vs. 0.73). Cross-cohort generalization was enhanced by up to 81%, addressing transferability challenges across different populations and technical protocols. Our approach provides a valuable framework for overcoming data limitations in microbiome research, with particular potential for the many clinical and intervention studies that operate with small cohorts.

---

## **Content**
This repo indcludes:
* embed_new_samples.py - a script to preprocess your samples, load the pretrained MAE-30% model, and generate sample representations
* model/mae30_encoder_full.pkl - the pretrained model
* model/model_species_reference.csv - a reference file with species names, to align the order of features (species) in your samples with what was used to train the model
* requirements.txt
* README.md
* LICENSE

---
## Contact

For any questions, please open an [issue on this repository](https://github.com/LiZa/MBEmbed/issues) or contact Liron Zahavi at liron.zahavi at weizmann.ac.il.

---
