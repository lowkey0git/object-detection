def ssd_pipeline(image):
    # Step 1: Feature extraction at multiple layers
    features = backbone_cnn(image)  # e.g., VGG

    predictions = []

    for feature_map in features:
        for cell in feature_map:
            for anchor_box in predefined_boxes:
                class_scores, box_offsets = predict(cell, anchor_box)
                predictions.append((class_scores, box_offsets))

    # Step 2: Apply Non-Max Suppression
    final_detections = non_max_suppression(predictions)
    return final_detections
