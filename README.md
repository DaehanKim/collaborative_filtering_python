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
random_user = list('abcdefghijklmnopqrstuvwxyz')
print(random_user)
# ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
random_item = [str(i) for i in range(20)]
print(random_item)
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
# {'a_0': 3.0374999999999996, 'a_7': 3.0374999999999996, 'a_8': 3.0374999999999996, 'a_10': 3.0374999999999996, 'a_11': 3.0837499999999998, 'a_12': 3.0374999999999996, 'a_13': 3.1137499999999996, 'a_14': 3.7937499999999997, 'a_15': 3.0374999999999996, 'a_16': 3.0374999999999996, 'a_17': 3.8387499999999997, 'a_18': 3.0374999999999996}

predicted, error = model.complete()
# predict ratings for all non-evaluated pairs considering all users and items 
# {'a_0': 1.789375, 'a_1': 1.589375, 'a_2': 2.391454950083955, 'a_4': 2.5843749999999996, 'a_5': 2.3582404453704133, ......}
```

#### SLIM

```
TBA
```