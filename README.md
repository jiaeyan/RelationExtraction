# RelationExtraction

1. Get on a department machine. Clone the GitHub repo, and get into that repo, and run following commands:
2. python run.py —train
3. python run.py —test 
4. sh mallet-maxent-classifier.sh -train  -model=model -gold=train.txt
5. sh mallet-maxent-classifier.sh -classify  -model=model -input=test.txt > test.tagged
6. python relation-evaluator.py data/rel-testset.gold test.tagged


Note: Step 5 takes long time, and please ignore the "ClassNotFound" error. 