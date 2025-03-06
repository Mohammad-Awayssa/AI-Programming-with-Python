#!/usr/bin/env python3
import argparse
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub

def process_image(image_path):
    # Load image with PIL
    pil_image = Image.open(image_path)
    # Convert the image to a NumPy array
    image_array = np.asarray(pil_image, dtype=np.float32)
    # Convert to a TensorFlow tensor
    image_tensor = tf.convert_to_tensor(image_array)
    # Resize the image to 224x224 pixels
    resized_image = tf.image.resize(image_tensor, (224, 224))
    # Normalize pixel values to [0, 1]
    normalized_image = resized_image / 255.0
    # Convert back to a NumPy array
    return normalized_image.numpy()

def predict(image_path, model, top_k=5):
    # Preprocess the image
    processed_image = process_image(image_path)
    # Add a batch dimension: model expects input shape (1, 224, 224, 3)
    expanded_image = np.expand_dims(processed_image, axis=0)
    # Make predictions
    predictions = model.predict(expanded_image)
    # Extract top K predictions using tf.math.top_k
    top_k_probs, top_k_classes = tf.math.top_k(predictions, k=top_k)
    # Convert tensors to NumPy arrays and extract from the batch
    probs = top_k_probs.numpy()[0]
    classes = top_k_classes.numpy()[0].astype(str)
    return probs, classes

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Predict flower name from an image using a trained model."
    )
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    parser.add_argument('saved_model', type=str, help='Path to the saved Keras model.')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes.')
    parser.add_argument('--category_names', type=str, default=None,
                        help='Path to JSON file mapping labels to flower names.')
    
    args = parser.parse_args()

    # Load the model with custom objects for KerasLayer
    model = tf.keras.models.load_model(
        args.saved_model,
        custom_objects={'KerasLayer': hub.KerasLayer},
        compile=False
    )
    
    # Get predictions from the model
    probs, classes = predict(args.image_path, model, top_k=args.top_k)
    
    # If a category names file is provided, load it and map indices to names
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        # If the JSON file is a list (indexed by integer),
        # convert the class indices (currently strings) to integers
        try:
            class_names = [cat_to_name[int(cls)] for cls in classes]
        except (ValueError, KeyError):
            # If the keys are strings, or conversion fails, try using them as strings
            class_names = [cat_to_name.get(cls, cls) for cls in classes]
    else:
        # If no mapping is provided, use the class indices
        class_names = classes

    # Print the predictions
    print("Predictions:")
    for prob, name in zip(probs, class_names):
        print(f"{name}: {prob*100:.2f}%")

if __name__ == '__main__':
    main()
