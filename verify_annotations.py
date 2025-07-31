import os

annot_dir = "processed_data/cattleface/train/annotations/"
for annot_file in os.listdir(annot_dir):
    with open(os.path.join(annot_dir, annot_file), 'r') as f:
        lines = f.readlines()
        if not lines:
            print(f"Empty annotation: {annot_file}")
        for line in lines:
            data = line.strip().split()
            if len(data) < 5:
                print(f"Invalid line in {annot_file}: {line.strip()}")
                continue
            try:
                class_id, x_center, y_center, width, height = map(float, data[:5])
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                    print(f"Non-normalized coordinates in {annot_file}: {line.strip()}")
            except ValueError:
                print(f"Non-numeric values in {annot_file}: {line.strip()}")
