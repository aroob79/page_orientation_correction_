import os
from collections import Counter
import matplotlib.pyplot as plt

def yolo_label_distribution(label_dir, class_names=None, plot=True):
    """
    Args:
        label_dir (str): path to YOLO label .txt files
        class_names (list): optional list of class names
        plot (bool): whether to plot distribution
    """

    label_counter = Counter()
    total_instances = 0

    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(label_dir, file)
            with open(file_path, "r") as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        label_counter[class_id] += 1
                        total_instances += 1

    print("\n📊 YOLO Label Distribution\n")
    print(f"Total instances: {total_instances}\n")

    for class_id, count in sorted(label_counter.items()):
        name = class_names[class_id] if class_names else f"Class {class_id}"
        percentage = (count / total_instances) * 100
        print(f"{name:<15} -> {count:>6} instances ({percentage:.2f}%)")

    # Optional bar plot
    if plot:
        labels = [
            class_names[i] if class_names else f"Class {i}"
            for i in label_counter.keys()
        ]
        counts = list(label_counter.values())

        plt.figure()
        plt.bar(labels, counts)
        plt.xticks(rotation=45)
        plt.title("YOLO Label Distribution")
        plt.ylabel("Number of Instances")
        plt.xlabel("Classes")
        plt.tight_layout()
        plt.show()

    return label_counter


label_dir = "/mnt/storage1/workspace/arobin/page_orientation/data/PAGE_ORIENTATION_DATA_WITH_4_CLASS/labels"
class_names = ["2nd", "4th", "down", "up"]

yolo_label_distribution(label_dir, class_names)
