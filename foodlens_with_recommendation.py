import cv2
import numpy as np
import tensorflow as tf
from food_recommendation import FoodPlaceRecommender
import os
import pandas as pd
import json

class FoodLensWithRecommendation:
    def __init__(self, model_path='model_indonesian_food (1).keras'):
        try:
            # Load the food recognition model
            print("\nInitializing FoodLens system...")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Looking for model at: {os.path.abspath(model_path)}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
                
            print("Loading model from:", model_path)
            self.model = tf.keras.models.load_model(model_path)
            
            # Print model summary to understand its structure
            print("\nModel Summary:")
            self.model.summary()
            
            # Get the number of output classes from the model
            last_layer = self.model.layers[-1]
            if isinstance(last_layer, tf.keras.layers.Dense):
                num_classes = last_layer.units
                print(f"\nNumber of output classes in model: {num_classes}")
            else:
                raise ValueError("Last layer is not a Dense layer with softmax activation")

            # Load class names from class_indices.json if available
            class_indices_path = 'class_indices.json'
            if os.path.exists(class_indices_path):
                with open(class_indices_path) as f:
                    class_indices = json.load(f)
                class_names = [None] * len(class_indices)
                for label, idx in class_indices.items():
                    class_names[idx] = label
                self.class_names = class_names
                print("Class names loaded from class_indices.json:", self.class_names)
            else:
                # fallback to dataset folder if class_indices.json does not exist
                dataset_path = 'dataset_indonesian_food'
                if os.path.exists(dataset_path):
                    self.class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
                    print("Class names loaded from dataset folder:", self.class_names)
                else:
                    raise ValueError("Tidak ditemukan class_indices.json maupun folder dataset_indonesian_food untuk mendapatkan class names.")
            
            # Make sure the number of class names matches the model output
            if len(self.class_names) != num_classes:
                raise ValueError(
                    f"Jumlah label makanan ({len(self.class_names)}) "
                    f"tidak sama dengan jumlah output kelas model ({num_classes})!\n"
                    "Pastikan dataset, class_indices.json, dan model cocok."
                )

            # Initialize the recommendation system
            print("\nInitializing recommendation system...")
            self.recommender = FoodPlaceRecommender()
            self.recommender.load_places_data('places_data.json')
            
            print("\nFoodLens system initialized successfully!")
                
        except Exception as e:
            print(f"\nError initializing FoodLens: {str(e)}")
            raise
        
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Preprocess image for model prediction"""
        try:
            print(f"\nPreprocessing image: {image_path}")
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read the image file")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            img = img.astype(np.float32) / 255.0
            print("Image preprocessing completed")
            return img
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")
        
    def recognize_food(self, image_path):
        """Recognize food in the image"""
        try:
            print(f"\nRecognizing food in image: {image_path}")
            img = self.preprocess_image(image_path)
            img = np.expand_dims(img, axis=0)
            
            # Get prediction
            print("Running model prediction...")
            predictions = self.model.predict(img, verbose=0)[0]
            predicted_class = np.argmax(predictions)
            
            # Print prediction details for debugging
            print(f"\nPrediction details:")
            print(f"Predicted class index: {predicted_class}")
            print(f"Number of classes in model: {len(predictions)}")
            print(f"Number of class names: {len(self.class_names)}")
            
            # Print all predictions for debugging
            print("\nAll predictions:")
            for idx, (name, prob) in enumerate(zip(self.class_names, predictions)):
                print(f"{name}: {prob:.2%}")
            
            # Print top 3 predictions for debugging
            top_3_idx = np.argsort(predictions)[-3:][::-1]
            print("\nTop 3 predictions:")
            for idx in top_3_idx:
                print(f"{self.class_names[idx]}: {predictions[idx]:.2%}")
            
            # Verify the predicted class index is valid
            if predicted_class >= len(self.class_names):
                raise ValueError(
                    f"Predicted class index {predicted_class} is out of range. "
                    f"Model has {len(predictions)} classes but class_names has {len(self.class_names)} items"
                )
                
            confidence = predictions[predicted_class]
            
            return {
                'food_name': self.class_names[predicted_class],
                'confidence': float(confidence),
                'top_predictions': [
                    {
                        'food_name': self.class_names[idx],
                        'confidence': float(predictions[idx])
                    }
                    for idx in top_3_idx
                ],
                'all_predictions': [
                    {
                        'food_name': name,
                        'confidence': float(prob)
                    }
                    for name, prob in zip(self.class_names, predictions)
                ]
            }
        except Exception as e:
            raise Exception(f"Error recognizing food: {str(e)}")
        
    def get_recommendations(self, food_name):
        """Get restaurant recommendations for the recognized food"""
        try:
            print(f"\nGetting recommendations for: {food_name}")
            recommendations = self.recommender.recommend_places(food_name)
            return recommendations
        except Exception as e:
            raise Exception(f"Error getting recommendations: {str(e)}")
        
    def process_image_and_recommend(self, image_path):
        """Process image and get both recognition and recommendations"""
        try:
            # Recognize food
            recognition_result = self.recognize_food(image_path)
            
            # Get recommendations
            recommendations = self.get_recommendations(
                recognition_result['food_name']
            )
            
            return {
                'recognition': recognition_result,
                'recommendations': recommendations
            }
        except Exception as e:
            raise Exception(f"Error processing image and getting recommendations: {str(e)}")

# Example usage
if __name__ == "__main__":
    try:
        # Initialize the system
        foodlens = FoodLensWithRecommendation()
        
        # Process an image and get recommendations
        image_path = "path_to_your_image.jpg"  # Replace with actual image path
        results = foodlens.process_image_and_recommend(image_path)
        
        # Print results
        print("\nFood Recognition Result:")
        print(f"Recognized Food: {results['recognition']['food_name']}")
        print(f"Confidence: {results['recognition']['confidence']:.2%}")
        
        print("\nTop 3 Predictions:")
        for pred in results['recognition']['top_predictions']:
            print(f"{pred['food_name']}: {pred['confidence']:.2%}")
        
        print("\nRecommended Places:")
        for place in results['recommendations']:
            print(f"\n{place['name']}")
            print(f"Cuisine: {place['cuisine']}")
            print(f"Rating: {place['rating']}")
            print(f"Features: {', '.join(place['features'])}")
    except Exception as e:
        print(f"Error: {str(e)}")

with open('class_indices.json') as f:
    class_indices = json.load(f)
class_names = [None] * len(class_indices)
for label, idx in class_indices.items():
    class_names[idx] = label 