# Master's Thesis: Using Feature Relevance to Control Neural Plasticity for Knowledge Preservation in Continual Learning - Chaymaa Zimame

This repository contains the data, models, configurations, methods, a cloned and partly modified LRP implementation, baseline training scripts, experiments and evaluation scripts developed as part of my master's thesis.

- **data/** – Dataset preparation and preprocessing
- **model/** – Model implementations
- **lrp_repo/** – Modified implementation of Layer-wise Relevance Propagation (LRP)
- **methods/** – Relevance and importance computation methods
- **tests_before_experiments/** – Preliminary tests conducted before the main experiments
- **experiments/** – Training scripts and experiment pipeline
- **configs/** – Configuration files
- **results/** – Baseline training results
- **results_experiments/** – Results of the conducted experiments
- **tests_experiments/** – Post-experiment evaluation and analysis

Clone the repository:

```bash
git clone <repository-url>
cd ma-chaymaa-zimame
```
Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---


## Acknowledgements: Third-Party Code
This project incorporates a modified version of the Layer-Wise Relevance Propagation (LRP) implementation from [[ECCV24] Layer-Wise Relevance Propagation with Conservation Property for ResNet](https://github.com/keio-smilab24/LRP-for-ResNet).

The original implementation is available in the referenced repository.
The code in `lrp_repo/` has been adapted for use in this project.
