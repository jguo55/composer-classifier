from matplotlib import pyplot as plt


losses = [1.2, 0.9, 0.5, 0.13, 0.23, 0.04, 0.01]
val_losses = [1.3, 0.8, 0.3, 0.2, 0.5, 0.9]

if len(val_losses) > 2:
    if val_losses[-1]  > val_losses[-3]:
        print("yes")
plt.plot(losses, label='loss')
plt.plot(val_losses)
plt.show()
