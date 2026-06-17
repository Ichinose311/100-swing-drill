'''
	11回目	第9章 85～89
		ファインチューニングで最適化するスコアは最後のepochのvalid accuracyもしくはlossではなく，すべてのepochの中での最良のvalid accuracyもしくはlossとすること
		学習率は1e-3～1e-6程度にすると良い
		OptimizerはAdam, AdamWあたりを使うと良い
		[深層学習モデルの学習のコツ]も参照すること
		学習にはTransformersのTrainerを使っても良い (使わなくても良い)
			使うと学習コードが非常に簡単に書ける
			参考：
				https://qiita.com/m__k/items/2c4e476d7ac81a3a44af
				https://qiita.com/nipo/items/44ce3aaf6acd4e2649d1
'''