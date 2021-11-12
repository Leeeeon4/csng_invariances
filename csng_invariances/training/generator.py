"""Generator models module."""


# TODO sort this file


gan = ComputeModel(generator_model, encoding_model)

optimizer = optim.Adam(generator_model.parameters())
loss_function = SelectedNeuronActivation()

running_loss = 0.0
for epoch in range(2000):
    optimizer.zero_grad()
    activations = gan(latent_tensor)
    loss = loss_function(activations, 5)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    if epoch % 200 == 0:
        print("[%d] loss: %.3f" % (epoch + 1, running_loss))
    running_loss = 0.0
