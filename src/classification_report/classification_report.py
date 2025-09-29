from log_classification.classifier import Classifier
from log_classification.preprocessor import Preprocessor
import numpy as np
from rich.console import Console
from rich.table import Table
import argparse

# TODO: Irgend eine bessere alternative für pfadangaben finden
# - eventuell einen "project path" in einer globalen konfiguration angeben und alle pfade relativ dazu betrachten
# - dort werden preprocessors, datasets, encoders, classifiers 



def condense_entries(data, threshold=0.05):
    result = []
    if not data:
        return result
    
    # Start with the first entry
    current_probs, (current_start, current_end) = data[0]
    current_class = np.argmax(current_probs)
    current_max = current_probs[current_class]
    count = 1
    
    for probs, (start, end) in data[1:]:
        class_id = np.argmax(probs)
        max_prob = probs[class_id]
        
        # Check if same class and max probs close enough
        if class_id == current_class and abs(max_prob - current_max) <= threshold:
            # merge
            current_probs += probs
            current_end = end
            count += 1
            current_max = (current_max * (count - 1) + max_prob) / count  # update average max
        else:
            # finalize current group
            avg_probs = current_probs / count
            result.append((current_class, avg_probs, (current_start, current_end)))
            
            # start new group
            current_probs = probs.copy()
            current_start, current_end = start, end
            current_class = class_id
            current_max = max_prob
            count = 1
    
    # finalize last group
    avg_probs = current_probs / count
    result.append((current_class, avg_probs, (current_start, current_end)))
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Log classifier.")
    parser.add_argument("--classifier-path", required=True, help="Path to the classifier file (e.g., model.keras).")
    parser.add_argument("--logfile-path", required=True, help="Path to the log file to classify.")
    parser.add_argument("--start-line", type=int, required=True, help="Start line number for classification.")
    parser.add_argument("--end-line", type=int, required=True, help="End line number for classification.")
    parser.add_argument("--condensation-threshold", type=float, default=0.05, help="Threshold for condensing entries.")

    args = parser.parse_args()

    # Load classifier
    classifier = Classifier.load(args.classifier_path)
    classes = classifier.pp.classes.values

    # Run prediction
    p, l = classifier.predict(args.logfile_path, args.start_line, args.end_line)
    condensed = condense_entries(list(zip(p, l)), threshold=args.condensation_threshold)

    # Formatting
    max_start_line = max([x[2][0] for x in condensed])
    max_end_line = max([x[2][1] for x in condensed])
    start_line_pad = len(str(max_start_line))
    end_line_pad = len(str(max_end_line))
    classes_pad = max([len(x) for x in classes])

    console = Console()

    # Create a table
    table = Table(title="Log Classification Result")

    header = ["Lines", "Most likely class", "Confidence", "2nd most likely class", "Confidence"]
    for h in header:
        table.add_column(h)

    # Add rows
    for class_id, probs, (start, end) in condensed:
        sorted_ids = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)

        row_style = "red" if classes[class_id] != "Normal" else "green"
        start_line = f"{start:>{start_line_pad}}"
        end_line = f"{end:>{end_line_pad}}"
        top1_class = f"{classes[class_id]:>{classes_pad}}"
        top1_confidence = f"{100*probs[class_id]:.2f}%"
        top2_class = f"{classes[sorted_ids[1]]:>{classes_pad}}"
        top2_confidence = f"{100*probs[sorted_ids[1]]:.2f}%"

        table.add_row(f"{start_line} - {end_line}", top1_class, top1_confidence, top2_class, top2_confidence, style=row_style)

    # Print table
    console.print(table)


if __name__ == "__main__":
    main()