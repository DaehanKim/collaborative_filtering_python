# matrix_completion_python
This repository implements various matrix completion methods.

### Supported models

 - User-based Collaborative Filtering

### Tutorial

```python
    random_user = ["{:.2f}".format(item) for item in np.random.randn(10)]
	# ['-0.84', '-1.06', '-0.95', '0.77', '2.56', '-0.13', '-0.51', '-0.42', '-0.09', '-0.88']
	random_item = ["{:.2f}".format(item) for item in np.random.randn(10)]
	# ['1.35', '1.81', '-1.60', '-1.05', '-0.89', '-0.02', '-0.49', '-1.24', '-0.13', '0.48']
	random_score_dict = {}
	for user in random_user :
		for item in random_item :
			if np.random.randn()<0 : continue
			random_score_dict[f"{user}_{item}"] = round(np.random.uniform()*5,2)
	''' random_score_dict -->
	{'-0.84_1.35': 1.54, '-0.84_1.81': 1.84, '-0.84_-1.05': 3.25, '-0.84_-0.89': 4.49, '-0.84_-0.13': 0.25, '-1.06_1.81': 3.86, '-1.06_-1.60': 3.96, '-1.06_-1.05': 3.47, '-1.06_-0.89': 1.9, '-1.06_-0.02': 2.36, '-1.06_-0.49': 1.06, '-1.06_-0.13': 2.84, '-1.06_0.48': 1.42, '-0.95_-0.89': 1.7, '-0.95_-0.02': 3.76, '-0.95_-1.24': 3.5, '-0.95_-0.13': 0.23, '0.77_1.35': 3.28, '0.77_0.48': 2.43, '2.56_1.35': 0.56, '2.56_1.81': 2.07, '2.56_-1.05': 1.22, '2.56_-0.02': 4.53, '2.56_-1.24': 0.31, '2.56_-0.13': 0.68, '2.56_0.48': 3.94, '-0.13_-1.05': 1.61, '-0.13_-0.89': 3.23, '-0.13_-0.02': 4.85, '-0.13_-1.24': 4.82, '-0.13_0.48': 0.76, '-0.51_1.81': 1.25, '-0.51_-1.05': 2.94, '-0.51_-1.24': 0.79, '-0.51_-0.13': 4.97, '-0.42_1.81': 1.65, '-0.42_-1.60': 4.15, '-0.42_-1.05': 4.61, '-0.42_-0.89': 3.43, '-0.42_-0.02': 2.04, '-0.42_-0.49': 1.93, '-0.42_0.48': 2.75, '-0.09_1.35': 4.77, '-0.09_-1.60': 2.69, '-0.09_-1.05': 4.57, '-0.09_-0.89': 4.3, '-0.09_-0.49': 4.23, '-0.09_-1.24': 0.35, '-0.09_-0.13': 2.74, '-0.09_0.48': 4.55, '-0.88_-1.60': 3.73, '-0.88_-0.89': 1.72, '-0.88_-0.49': 3.67, '-0.88_-1.24': 0.39}
	'''


	model = CF(user_list = random_user, 
		item_list = random_item, 
		score_dict = random_score_dict)

	predicted, error = model.complete()
	print(predicted)
	''' Prediction in terms of user_item pairs (in terms of internal ids)!
	{'0_2': 2.274, '0_5': 2.274, '0_6': 2.274, '0_7': 2.274, '0_9': 2.274, '1_0': 2.60875, '1_7': 2.60875, '2_0': 2.2975000000000003, '2_1': 2.2975000000000003, '2_2': 2.2975000000000003, '2_3': 2.2975000000000003, '2_6': 2.2975000000000003, '2_9': 2.2975000000000003, '3_1': 2.855, '3_2': 2.855, '3_3': 2.855, '3_4': 2.855, '3_5': 2.855, '3_6': 2.855, '3_7': 2.855, '3_8': 2.855, '4_2': 1.9014285714285712, '4_4': 1.9014285714285712, '4_6': 1.9014285714285712, '5_0': 3.054, '5_1': 3.054, '5_2': 3.054, '5_6': 3.054, '5_8': 3.054, '6_0': 2.4875, '6_2': 2.4875, '6_4': 2.4875, '6_5': 2.4875, '6_6': 2.4875, '6_9': 2.4875, '7_0': 2.937142857142857, '7_7': 2.937142857142857, '7_8': 2.937142857142857, '8_1': 3.525, '8_5': 3.525, '9_0': 2.3775000000000004, '9_1': 2.3775000000000004, '9_3': 2.3775000000000004, '9_5': 2.3775000000000004, '9_8': 2.3775000000000004, '9_9': 2.3775000000000004}
	'''
	print(error)
	''' Errors in terms of user_item pairs (in terms of internal ids)!
	{}
	'''
```