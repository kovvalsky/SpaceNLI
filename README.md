# SpaceNLI
Natural Language Inference problems specialized for spatial semantics &amp; reasoning.

The NLI problem patterns and automatically generated NLI sample problems are under `dataset/`.  

# Generating data
The dataset of NLI sample problems is generated from the patterns (using the mini world with selection restrictions and a toy grammar) with the following command:

```
python/generate_nli.py --sr config/selection_restriction.yaml --gr python/base_grammar.fcfg dataset/problem_patterns.xml -v --out dataset/160x200.json --out-max-seed 200
```
Note that changes to the selection restriction file can result in a different sample dataset as the fixed number of problems (e.g., 200) per pattern are randomly drawn from the all possible samples per pattern.  



# Paper

Abzianidze, L., Zwarts, J., Winter, Y. (2023). **SpaceNLI: Evaluating the Consistency of Predicting Inferences In Space.** In Proceedings of the 4th Natural Logic Meets Machine Learning Workshop (NALOMA IV).
