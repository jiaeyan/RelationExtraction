# RelationExtraction

-- data_prep.py: reads and loads necessary data for the classifier.

-- features.py: contains "Mention" and "MentionPair" classes to do feature engineering.

-- run.py: contains the main function to run the whole system.

-- data folder: contains original train/dev/test file, and also extra knowledge and tree data.

-- model/train.txt/test.txt/test.tagged: outputs of the system with best performance.

Please follow the instructions below to get results. The final score should be 42.89%.
1. Get on a department machine. 
2. Clone the GitHub repo: https://github.com/jiaeyan/RelationExtraction , and get into that repo, and run following commands (if you would like to directly check the final performance, please do STEP 7):
3. python run.py --task train  # to generate train feature file
4. python run.py --task test   # to generate test feature file
5. sh mallet-maxent-classifier.sh -train  -model=model -gold=train.txt
6. sh mallet-maxent-classifier.sh -classify  -model=model -input=test.txt > test.tagged
7. python relation-evaluator.py data/rel-testset.gold test.tagged

Note: Step 6 takes long time, and please ignore the "ClassNotFound" error. 