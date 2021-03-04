# Categorical Variables

1. [Encoding Categorical Variables](#Encoding_Categorical_Variables)
2. [Dealing with Large Categorical Variables](#Dealing_with_Large_Categorical_Variables)

---

## Encoding Categorical Variables


- One-Hot Encoding
	- Each represtns a possible category
	- A  bit redundant
	-  Missing data can be encoded as the all-zeros vector
	- `sklearn.preprocessing.OneHotEncoder`, `torch.nn.functional.one_hot`, `pandas.get_dummies`

		| |e1|e2|e3|
		|---|---|---|---|
		|Dog|1|0|0|
		|Cat|0|1|0|
		|Fox|0|0|1|
	- e1 + e2 +e3 ... +ek = 1
	-  k possible categories


- [Dummy Coding](https://en.wikiversity.org/wiki/Dummy_variable_(statistics))
	-  k possible categories, but use k-1 to represent
	-   Cannot easily handle missing data, since the all-zeros vector is already mapped to the reference category.
	-  `pandas.get_dummies`
		
		| |e1|e2|
		|---|---|---|
		|Dog|1|0|
		|Cat|0|1|
		|Fox|0|0|
		
	- e3 => (0,0)



-  Effect Coding
	-   Similar to dummy coding, with the difference that the reference category is now represented by the vector of all –1’s.
	-    In linear regression models that are even simpler to interpret.
	-    Expensive


				
			| |e1|e2|
			|---|---|---|
			|Dog|1|0|
			|Cat|0|1|
			|Fox|-1|-1|
		
	- e3 => (-1, -1)

---

## Dealing with Large Categorical Variables

 

**Existing solutions can be:**

1. Do nothing fancy with the encoding. Use a simple and cheap model
2. Compress the features
	- Feature Hashing
		-  A [hash function](https://en.wikipedia.org/wiki/Hash_function) is a deterministic function that maps a potentially unbounded integer to a finite integer range [1, k]  hash table
		-   A [uniform hash function](https://en.wikipedia.org/wiki/Hash_function#Uniformity) ) ensures that roughly the same number of numbers are mapped into each of the m bins.
		
		```python
		def hash_feature(data_list, k):
			output = np.ones(k) #k categories
			for data in data_list :
				index = hashfunc(data) % k
				output[index] += 1
			return output
				
		```
	- Bin Counting
		-  Use the conditional probability of the target under that value
		-   Rare categories =>  back-off method