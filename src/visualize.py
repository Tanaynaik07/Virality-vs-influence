import matplotlib.pyplot as plt

def plot_results(results):
    """
    Plot bar chart of cascade sizes.
    """
    models = list(results.keys())
    sizes = list(results.values())
    
    plt.figure(figsize=(7, 5))
    plt.bar(models, sizes, color=["skyblue", "orange", "green"])
    plt.ylabel("Cascade Size (users)")
    plt.title("Comparison of IC, LT, and Hybrid Models")
    plt.show()
