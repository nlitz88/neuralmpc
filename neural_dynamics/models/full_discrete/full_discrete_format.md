# Fully-Learned Discrete Model

> model = nn.Sequential(
>    
>    nn.Linear(in_features=7, out_features=128),
>
>    nn.ReLU(),
>
>    nn.Linear(in_features=128, out_features=5))

Model Input:

`[uk, xk] == [u_a, u_steer, v_long, v_tran, psi, x, y]`

Model Output:

`[xk+1] == [v_long, v_tran, psi, x, y]`
