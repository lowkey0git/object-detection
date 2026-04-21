def yolo_forward_pass(image):
    grid = split_image_into_grid(image, S=7)
    features = conv_layers(grid)  # CNN extracts features
    output = fully_connected(features)  # Predict (x, y, w, h, conf, classes)

    final_predictions = post_process(output)  # Confidence threshold + NMS
    return final_predictions
