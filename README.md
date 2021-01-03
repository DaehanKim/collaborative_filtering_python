# Collaborative Filtering in python
This repository implements a Collaborative Filtering algorithm.

### Supported models

 - User-based Collaborative Filtering
   - Uses pearson correlation coefficient to measure similarity between users.
     - significance level is measured by p-value assuming normal distributions of both users ' scores. If p-value is larger than 0.1, computed similarity is regarded as invalid and assign 0.0 as a similarity.
     - If a score matrix(dictionary) is sparse, this algorithm is likely to just assign average score of a user's ratings as a predicted rating.
     - Specifically, predicted rating (<img src="https://render.githubusercontent.com/render/math?math=R_{ij}">) is the sum of mean rating of user i (<img src="https://render.githubusercontent.com/render/math?math=\overline{R_{i}}">) and the weighted average of neighbor users' rating (<img src="https://render.githubusercontent.com/render/math?math=N_i = \{R_{i n_1}, R_{i n_2},..., R_{i n_M}\}">) where M is predefined as 'num_neighbors'. Weights for each ratings (<img src="https://render.githubusercontent.com/render/math?math=S_{i} = \{ S_{i n_1}, ..., S_{i n_M} \}">) are predefined similarity measures (in this case, pearsonR). For details, see [this post](https://www.geeksforgeeks.org/user-based-collaborative-filtering/).
   - similarity measures currently supported are 'pearsonR' or 'pearsonR+'
     - pearsonR : measures pearson correlation between user and neighbor's ratings of common items. Predicted ratings could be negative.
     - pearsonR+ : uses pearsonR but disregards negative correlation(-1~0) and set it to zero in such cases. This ensures predicted ratings are within proper ranges.


### Usage

```python
random_user = list('abcdefghijklmnopqrstuvwxyz')
# ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
random_item = [str(i) for i in range(20)]
# ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']

random_score_dict = {}
for user in random_user :
  for item in random_item :
    # rating sparsity is 50%
    if np.random.uniform()<0.5 : continue
    # rating range is in [0,5]
    random_score_dict[f"{user}_{item}"] = round(np.random.uniform()*5,2)

model = CF(
user_list = random_user, 
item_list = random_item, 
score_dict = random_score_dict,
sim_metric = "pearsonR+",
method="user_based",
num_neighbors=5
)

predicted, error = model.complete_for(['a_{}'.format(i) for i in range(20)])
# predict all ratings for user 'a'
# {'a_6': 1.8476923076923077, 'a_8': 1.8476923076923077, 'a_10': 1.8476923076923077, 'a_12': 1.8476923076923077, 'a_13': 1.8476923076923077, 'a_15': 1.8476923076923077, 'a_19': 1.8476923076923077}

predicted, error = model.complete()
# predict ratings for all non-evaluated pairs considering all users and items 
# {'a_6': 1.8476923076923077, 'a_8': 1.8476923076923077, 'a_10': 1.8476923076923077, 'a_12': 1.8476923076923077, 'a_13': 1.8476923076923077, 'a_15': 1.8476923076923077, 'a_19': 1.8476923076923077, 'b_1': 3.211642392121491, 'b_2': 3.55875, 'b_6': 4.00375, 'b_7': 1.6737499999999998, 'b_10': 2.7856227678465375, 'b_11': 3.24875, 'b_12': 3.1574999999999998, 'b_13': 3.01875, 'b_14': 2.57375 ...}
```

### Disclaimer

Please use this implementation as a casual reference. For academic or industrial use, you might need to modify formulae as your work requires. 