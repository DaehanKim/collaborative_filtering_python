# matrix_completion_python
This repository implements various matrix completion methods.

### Supported models

 - User-based Collaborative Filtering
   - Uses pearson correlation coefficient to measure similarity between users.
     - significance level is measured by p-value assuming normal distributions of both users ' scores.
     - If p-value is larger than 0.1, computed similarity is regarded as invalid and assign 0.0 as a similarity.
     - If a score matrix(dictionary) is sparse, this algorithm results in just assigning average score of a user's ratings as a predicted rating. 
   - similarity measures currently supported are 'pearsonR' or 'pearsonR+'
     - pearsonR : measures pearson correlation between user and neighbor's ratings of common items. Predicted ratings could be negative.
     - pearsonR+ : uses pearsonR but disregards negative correlation(-1~0) and set it to zero in such cases. This ensures predicted ratings are positive.

  - SLIM algorithm
    - TBA

### Usage

#### Collaborative Filtering

```python
random_user = list("ABCDEFGHIJ")
# ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
random_item = list("abcdefghij")
# ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
random_score_dict = {}

# Assign random ratings ranging from 0 to 5
for user in random_user :
	for item in random_item :
		if np.random.randn()<0 : continue
		random_score_dict[f"{user}_{item}"] = round(np.random.uniform()*5,2)
		
model = CF(user_list = random_user, 
	item_list = random_item, 
	score_dict = random_score_dict)

predicted, error = model.complete()
print(predicted)
''' Prediction in terms of user_item pairs!
{'A_a': 2.92, 'A_e': 1.3099999999999998, 'A_f': 2.92, 'A_g': 2.92, 'A_j': 2.92, 'B_a': 2.2616666666666667, 'B_b': -2.208333333333333, 'B_c': 2.2616666666666667, 'B_e': 0.6516666666666666, 'C_a': 3.524, 'C_c': -0.8360000000000003, 'C_f': -0.7060000000000004, 'C_g': 0.5640000000000001, 'C_j': 3.294, 'D_a': 2.4539999999999997, 'D_b': 2.4539999999999997, 'D_f': 2.4539999999999997, 'D_i': 2.4539999999999997, 'D_j': 2.4539999999999997, 'E_b': 3.686666666666667, 'E_c': 3.686666666666667, 'E_d': 3.686666666666667, 'E_e': 3.686666666666667, 'E_f': 3.686666666666667, 'E_g': 3.686666666666667, 'E_h': 3.686666666666667, 'F_c': 1.7625, 'F_g': 1.7625, 'G_b': 2.65, 'G_d': 2.65, 'G_i': 2.65, 'G_j': 2.65, 'H_a': 2.38, 'H_b': 2.38, 'H_c': 2.38, 'H_e': 2.38, 'H_g': 2.38, 'H_i': 2.38, 'I_a': 3.0933333333333337, 'I_b': 3.0933333333333337, 'I_d': 3.0933333333333337, 'I_i': 3.0933333333333337, 'J_b': 1.875, 'J_d': 1.875, 'J_e': 1.875, 'J_g': 1.875, 'J_i': 1.875, 'J_j': 1.875}'''
print(error)
''' Errors in terms of user_item pairs!
{}
'''
```

#### SLIM

```
TBA
```