# pretrained_pinn/

- `pretrained_pinn.pth` – `state_dict` of the PINN (`pinnvsmech.pinn.NN_free_params.NN`).

Architecture: input `[t, R0, Delta]`, 5 hidden layers of 100 neurons with Tanh, 96 outputs (6 compartments x 16 age classes) with Sigmoid. Build it with `NN(activation=nn.Tanh, dimensions=[3, 100, 100, 100, 100, 100, 96])` and load it with `model.load_state_dict(torch.load(path))`.

Training: 33 x 33 grid on R0 in [1.3, 3.0] and Delta in [7, 35], weekly, 1 infected per age class; collocation set 32 x 32, daily; Adamax, lr 1e-3, batch size 100, 5000 epochs. ODE loss weight as in the file name (100). See `notebooks/02_pinn_training.ipynb`.
