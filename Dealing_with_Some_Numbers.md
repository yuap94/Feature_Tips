# Dealing with Some Number

1. [Counts](#Counts)
2. [Log Transformation](#Log_Transformation)
3. [Scaling & Normalization](#Scaling_&_Normalization)
4. [Interaction Features](#Interaction_Features)
5. [Feature Selection](#Feature_Selection)

[Back to TOP](README.md)

---
## Counts
=>  Histogram
 - Fast plot: [`pandas.DataFrame.hist`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.hist.html#pandas.DataFrame.hist)
	```python
	import pandas as pd
	df = pd.read_csv("somedata.csv")
	df['some feature'].hist()
	```
 - Binarization : 
	 - Sacrifice information and make data more regularized. 
	 - Makes model more robust and prevent overfitting.  
	 - Can be applied to both categorical and numerical data.
	  

- Quantization or Binning
	-  Fixed-width binning : with specific numeric range
	-  Quantile binning :  
		-  Dealing some large gaps in the counts.
		-  Quantiles are values that divide the data into equal portions.
 
---

## Log Transformation

- Only positive data
- The distribution becomes more approximate to normal and  decreases the effect of the outliers
	- Dealing with positive numbers with a heavy-tailed distribution.  It compresses the long tail in the high end of the distribution into a shorter tail, and expands the low end into a longer head.
-  EX: [Power Transforms](https://en.wikipedia.org/wiki/Power_transform): Generalization of the Log Transform
	-   variance-stabilizing transformations.
	-   Box-Cox transform: A simple generalization of both the square root transform and the log transform   

---

## Scaling & Normalization

- Min-Max Scaling
	```math
	x = (x - min(x)) / (max(x) - min(x))
	```
- Standardization (Variance Scaling)

	```math
	x = (x - mean(x)) / sqrt(var(x))
	```

-  L2 Normalization (Euclidean norm)
	```python
	x = x / np.linalg.norm(x)
	```

---

## Interaction Features
 
 - Features interact with each other in a prediction model

- A simple pairwise interaction feature is the product of two features.
- [More info](https://christophm.github.io/interpretable-ml-book/interaction.html)

---

## Feature Selection

- Filtering 
	- Remove ones that are unlikely to be useful for the model.
	- Cost cheap
- Wrapper methods
	- Subsets of features
	- Cost expensive
- Embedded methods
	-  Perform feature selection as part of the model training process.
	-  Ex:  Decision tree
	-  Cost OK