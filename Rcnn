def rcnn_pipeline(image):
    # Step 1: Generate region proposals
    region_proposals = selective_search(image)

    results = []

    for region in region_proposals:
        resized = resize(region, (224, 224))

        # Step 2: Feature extraction
        features = cnn(resized)

        # Step 3: Classification
        class_label = svm_classifier(features)

        # Step 4: Box regression
        box = bbox_regressor(features)

        results.append((class_label, box))

    # Post-processing: Remove overlaps
    final_detections = non_max_suppression(results)
    return final_detections
