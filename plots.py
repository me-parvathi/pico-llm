import matplotlib.pyplot as plt


def plot_batch_losses(loss_dict, save_path=None):
    """
    Plot training loss per batch for each model.
    
    Args:
        loss_dict: Dictionary mapping model names to lists of loss values
                  Format: {model_name: [loss1, loss2, ...]}
        save_path: Optional path to save the figure. If None, figure is not saved.
    """
    plt.figure(figsize=(10, 5))
    
    for model_name, losses in loss_dict.items():
        plt.plot(losses, label=model_name)
    
    plt.xlabel('Batch Index')
    plt.ylabel('Loss')
    plt.title('Training Loss per Batch')
    plt.legend()
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path)
    
    plt.close()

