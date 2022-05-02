

# Codebase Ideas & To Do
1) Check to see where any memory leaks may exist (so scaling up to more workers is feasible)
2) Consider moving the broadcast to after the averaging

The irecv was kept in order to ensure if more messages came in, then they could be received.
If we start to use the asynchronous update method, then we move to Recv.