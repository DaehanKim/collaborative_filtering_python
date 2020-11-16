import numpy as np
from tqdm import tqdm
import sys
import math 
from scipy import stats


class CF:
	def __init__(self, 
		user_list : list, 
		item_list: list, 
		score_dict : dict, 
		sim_metric="pearsonR", 
		method = 'user_based', 
		num_neighbors = 3,
		verbose = False):
		'''
		This implements CF method based on `https://www.geeksforgeeks.org/user-based-collaborative-filtering/?ref=rp`
		Params : 
		  sim_metric (str) : must be one of "pearsonR", "pearsonR+"
		'''
		self.sim_metric = sim_metric
		assert sim_metric in ['pearsonR','pearsonR+'], "sim_metric must be in 'pearsonR' or 'pearsonR+'"
		self.method = method
		self.verbose = verbose
		self.num_neighbors = num_neighbors
		self._init(user_list = user_list, 
			item_list = item_list,
			score_dict = score_dict)

	def _init(self, user_list, item_list, score_dict):
		'''
		Initialize model attributes
		Params : user_list(list), 
		item_list(list), 
		score_dict (dict[`user`_`item`, rating_score]) : contains pair, score information
		Returns : None
		'''
		self.user_list = user_list
		self.user2id = {user : idx for idx, user in enumerate(user_list)}
		self.item_list = item_list 
		self.item2id = {item : idx for idx, item in enumerate(item_list)}

		self.score_dict_in_id = {self._convert_pair2id(k):v for k,v in score_dict.items()}

		# get non-evaluated (user, item) pairs
		self.non_eval_pairs = [f"{user_id}_{item_id}" for user_id in range(len(user_list)) for item_id in  range(len(item_list)) if f"{user_id}_{item_id}" not in self.score_dict_in_id]
		
		self.user_mean_score = {}
		for i in range(len(self.user_list)):
			keys = [item for item in self.score_dict_in_id.keys() if item.split("_")[0] == str(i)]
			self.user_mean_score[i] = None if not keys else np.array([self.score_dict_in_id[key] for key in keys]).mean()

	def _convert_pair2id(self,name_pair):
		user,item = name_pair.split("_")
		return f"{self.user2id[user]}_{self.item2id[item]}"

	def _convert_pair2name(self, id_pair):
		user_id, item_id = id_pair.split("_")
		user_id, item_id = int(user_id), int(item_id)
		return f"{self.user_list[user_id]}_{self.item_list[item_id]}"

	def _compute_sim(self):
		'''
		Compute similarity dictionary.
		Params : 
		  self
		Returns : 
		  similarity_dict (dict[user_id, dict[user_id, similarity]])
		'''

		# initialize similarity with itself to 1
		self.sim_dict = {user_id : {user_id : 1.} for user_id, _ in enumerate(self.user_list)}
		
		for i, user_1 in tqdm(enumerate(self.user_list), desc="construct similarity matrix", total=len(self.user_list)):
			for j, user_2 in enumerate(self.user_list):
				if i >= j : continue
				i_items = [(pair.split('_')[1], score) for pair, score in self.score_dict_in_id.items() if pair.split('_')[0] == str(i)]
				j_items = [(pair.split('_')[1], score) for pair, score in self.score_dict_in_id.items() if pair.split('_')[0] == str(j)]

				i_scores, j_scores = [], []
				for common_item in set([pair[0] for pair in i_items]).intersection([pair[0] for pair in j_items]):
					i_scores.append(self.score_dict_in_id[f"{i}_{common_item}"])
					j_scores.append(self.score_dict_in_id[f"{j}_{common_item}"])
					try : 
						sim, p_val = stats.pearsonr(i_scores, j_scores)
						if self.sim_metric == "pearsonR+":
							sim = max(0, sim)
						if p_val < 0.1 : 
							self.sim_dict[i][j] = sim
							self.sim_dict[j][i] = sim
						else:
							raise (ValueError, "{i} and {j} : too little samples!")
					except ValueError:
						if self.verbose : print(f'{i} and {j} : Less than 2 sample pairs to compute correlation!')
						self.sim_dict[i][j] = 0 # assume no correlation
						self.sim_dict[j][i] = 0
					except : 
						if self.verbose : print(f"Error in pearsonr : {sys.exc_info()[0]}")
						self.sim_dict[i][j] = 0 # assume no correlation
						self.sim_dict[j][i] = 0

					

	def _score_pair(self, user_id, item_id):
		'''
		Compute score for a single (user, item) pair.
		Params : 
		Returns : 
		'''
		sim_list = list(self.sim_dict[user_id].items())
		sim_list.sort(key=lambda x:-abs(x[1]))

		pred = 0.
		sim_sum = 0.
		num_neighbor = 0
		for neighbor_user_id, sim_score in sim_list:
			if self.num_neighbors <= num_neighbor : break
			if f"{neighbor_user_id}_{item_id}" not in self.score_dict_in_id: continue
			pred += self.score_dict_in_id[f"{neighbor_user_id}_{item_id}"] * sim_score
			sim_sum += abs(sim_score)
			num_neighbor += 1
		# user's rating average has the same impact as aggregated neighbors' on the weighted average
		if sim_sum != 0 : 
			pred /= sim_sum*2
			pred += self.user_mean_score[user_id]*0.5
		else : 
			pred += self.user_mean_score[user_id] 
		return pred


	def complete(self):
		'''
		Compute scores for non-evaluated pairs.
		Params : 
		Returns : 
		'''
		self._compute_sim()
		self._pred_scores = {}
		_error ={}
		for _pair in tqdm(self.non_eval_pairs, desc = "score unevaluted pairs"):
			try : 
				self._pred_scores[_pair] = self._score_pair(*[int(item) for item in _pair.split('_')])
			except : 
				_error[_pair] = sys.exc_info()[0]

		named_pred_scores = {self._convert_pair2name(k) : v for k,v in self._pred_scores.items()}
		return named_pred_scores, _error



if __name__ == "__main__":
	random_user = list("ABCDEFGHIJ")
	print(random_user)
	random_item = list("abcdefghij")
	print(random_item)
	random_score_dict = {}
	for user in random_user :
		for item in random_item :
			if np.random.randn()<0 : continue
			random_score_dict[f"{user}_{item}"] = round(np.random.uniform()*5,2)

	model = CF(user_list = random_user, 
	item_list = random_item, 
	score_dict = random_score_dict,
	sim_metric = "pearsonR+")

	predicted, error = model.complete()
	print(predicted)
