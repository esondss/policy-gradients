# Policy Gradients
### A collection of Policy Gradients methods in Deep RL


![openaigym](https://user-images.githubusercontent.com/53110326/81946457-f3702b80-9631-11ea-9afd-064e8b7a1ff3.gif)


PG's learning is smoother than Q-learning, but update can only be mad by the end of an episode. We compare two version of REINFORCE: One ues z-score as bias and the other one uses state-value V(s). The figure below shows the total rewards from 50 iterations, each iteration has 10 episodes. With parameters being the same, their results are very similar, but state-value-biased REINFORCE seems to be less oscillated. 
![fig](https://user-images.githubusercontent.com/53110326/82118978-34407f80-97ad-11ea-9582-c2ab63257b23.png)
