def compute_ods(inputs, targets, threshold=0.5):
    label = targets.float()
    mask = targets.clone()
    mask[label == 1.] = 1.0
    mask[label == 0.] = 0
    mask[label == 2.] = 0

    # Binarize the masks using a threshold
    mask = (mask > threshold).byte()
    inputs = (inputs > threshold).byte()

    # Calculate True Positives, False Positives, and False Negatives
    true_positives = (mask * inputs).sum().float()
    false_positives = (inputs - (mask * inputs)).sum().float()
    false_negatives = (mask - (mask * inputs)).sum().float()

    # Calculate Precision and Recall
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)

    # Calculate F1-score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    # return precision, recall, f1_score
    return f1_score
