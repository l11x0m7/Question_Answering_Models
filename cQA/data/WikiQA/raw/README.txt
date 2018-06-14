Microsoft Research WikiQA Corpus

-------------------------------------------------------------------------------------------------------
Contact Persons: Scott Wen-tau Yih        Microsoft Research (scottyih@microsoft.com)
				 Yi Yang                  Georgia Tech (yangyiycc@gmail.com)
-------------------------------------------------------------------------------------------------------

We release both WikiQA dataset and python evaluation code for the Answer Triggering task used in the paper: 
WikiQA: A Challenge Dataset for Open-Domain Question Answering.

Please cite it if you use this dataset.

@InProceedings{YangYihMeek:EMNLP2015:WikiQA,
  author    = {Yang, Yi  and  Yih, Wen-tau  and  Meek, Christopher},
  title     = {{WikiQA}: {A} Challenge Dataset for Open-Domain Question Answering},
  booktitle = {Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  month     = {September},
  year      = {2015},
  address   = {Lisbon, Portugal},
  publisher = {Association for Computational Linguistics}
}

-------------------------------------------------------------------------------------------------------


Version 1.0: August 25, 2015

-------------------------------------------------------------------------------------------------------
SUMMARY

The WikiQA corpus is a new publicly available set of question and sentence pairs, collected and annotated for research on open-domain question answering. In order to reflect the true information need of general users, we used Bing query logs as the question source. Each question is linked to a Wikipedia page that potentially has the answer. Because the summary section of a Wikipedia page provides the basic and usually most important information about the topic, we used sentences in this section as the candidate answers. With the help of crowdsourcing, we included 3,047 questions and 29,258 sentences in the dataset, where 1,473 sentences were labeled as answer sentences to their corresponding questions. More detail of this corpus can be found in our EMNLP-2015 paper, "WikiQA: A Challenge Dataset for Open-Domain Question Answering" [Yang et al. 2015]. In addition, this download also includes the experimental results in the paper, an evaluation script for judging the "answer triggering" task, as well as the answer phrases labeled by the authors of the paper. 

-------------------------------------------------------------------------------------------------------
LIST OF FILES

LICENSE.pdf -- Microsoft Research Data License Agreement for Microsoft Research WikiQA Corpus

Guidelines_Phase1.pdf   -- The crowdsourcing labeling guidelines for the main annotation task
Guidelines_Phase2.pdf   -- The crowdsourcing labeling guidelines for the verification task

WikiQA.tsv                                          -- Original data
WikiQA-train.tsv, WikiQA-dev.tsv, WikiQA-test.tsv   -- train/dev/test split of WikiQA.tsv used in this paper
WikiQA-train.txt, WikiQA-dev.txt, WikiQA-test.txt   -- slightly processed data (tokenization)
WikiQA-train.ref, WikiQA-dev.ref, WikiQA-test.ref   -- reference files in TREC evaluation format
WikiQA-dev-filtered.ref, WikiQA-test-filtered.ref   -- removed all questions without correct answers in the candidate sentence sets (for Answer Sentence Selection evaluation)

WikiQASent.pos.ans.tsv   -- Questions with answer sentence annotated with answer phrases

Evaluation Code:
eval.py            -- python evaluation code for evaluation of the Answer Triggering task

-------------------------------------------------------------------------------------------------------
DATA FORMAT

WikiQA.tsv, WikiQA-train.tsv, WikiQA-dev.tsv, WikiQA-test.tsv	
	-- See the header row (i.e., the first line of the file)
WikiQA-train.txt, WikiQA-dev.txt, WikiQA-test.txt   
	-- Tokenized question, Tokenized candidate answer sentence, Label
WikiQASent.pos.ans.tsv
	-- See the header row (i.e., the first line of the file)
	-- The last three columns are answer phrases labeled by the first, second and third annotator.  If the field is an empty string, then it means that the annotator did not label this sentence.  If the string is NO_ANS, then it means that the annotator does not think the sentence is a correct answer to the question.

-------------------------------------------------------------------------------------------------------
EVALUATION

Answer Triggering evaluation example
python eval.py WikiQA-test.ref emnlp-table/WikiQA.CNN-Cnt.test.rank

-------------------------------------------------------------------------------------------------------
