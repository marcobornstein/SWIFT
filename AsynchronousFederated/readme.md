

# Codebase Ideas & To Do
1) Fix the data partitioning for non-iid (only create data for each worker, not all)
2) Wrap the communication in Train.py into a new function (clean the code)
3) Create a new Util.py with the average meter and the accuracy computations
4) Examine if the learning rate function outside of the training loop improves performance
5) Fix the model.eval() issue in ModelAvg.py (send the consensus model only after the 5th epoch?)