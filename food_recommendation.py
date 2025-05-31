import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os

class FoodPlaceRecommender:
    def __init__(self):
        self.places_data = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.place_vectors = None
        self.place_names = []
        
    def load_places_data(self, data_path):
        """Load places data from JSON file"""
        print(f"\nLoading places data from: {data_path}")
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                self.places_data = json.load(f)
            print(f"Successfully loaded {len(self.places_data)} places")
            self._prepare_vectors()
        else:
            print(f"Error: Data file not found at {data_path}")
            
    def _prepare_vectors(self):
        """Prepare TF-IDF vectors for places"""
        print("\nPreparing TF-IDF vectors...")
        place_descriptions = []
        self.place_names = []
        
        for place in self.places_data:
            # Combine relevant features for vectorization
            description = f"{place['name']} {place['cuisine']} {' '.join(place['features'])}"
            place_descriptions.append(description)
            self.place_names.append(place['name'])
            
        self.place_vectors = self.vectorizer.fit_transform(place_descriptions)
        print(f"Created vectors for {len(self.place_names)} places")
        
    def recommend_places(self, food_item, max_recommendations=5):
        """
        Recommend places based on food item
        
        Args:
            food_item (str): The recognized food item
            max_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of recommended places with their details
        """
        print(f"\nGetting recommendations for: {food_item}")
            
        if not self.places_data:
            print("Error: No places data available")
            return []
            
        # Create query vector
        query = f"{food_item}"
        print(f"Search query: {query}")
            
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(query_vector, self.place_vectors).flatten()
        
        # Get top recommendations
        top_indices = similarity_scores.argsort()[-max_recommendations:][::-1]
        
        recommendations = []
        for idx in top_indices:
            place = self.places_data[idx]
            recommendations.append({
                'name': place['name'],
                'cuisine': place['cuisine'],
                'location': place.get('location', ''),
                'rating': place['rating'],
                'features': place['features'],
                'similarity_score': float(similarity_scores[idx])
            })
            
        print(f"Found {len(recommendations)} recommendations")
        for rec in recommendations:
            print(f"- {rec['name']}")
            
        return recommendations

# Example usage
if __name__ == "__main__":
    # Sample places data
    sample_data = [
        {
            "name": "Warung Padang Sederhana",
            "cuisine": "Padang",
            "location": "Jakarta Selatan",
            "rating": 4.5,
            "features": ["Nasi Padang", "Rendang", "Ayam Pop"]
        },
        {
            "name": "Sate Khas Senayan",
            "cuisine": "Indonesian",
            "location": "Jakarta Pusat",
            "rating": 4.3,
            "features": ["Sate", "Nasi Goreng", "Sop Kambing"]
        }
    ]
    
    # Save sample data
    with open('places_data.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    # Initialize and test recommender
    recommender = FoodPlaceRecommender()
    recommender.load_places_data('places_data.json')
    
    # Test recommendations
    recommendations = recommender.recommend_places("Rendang")
    print("\nRecommended places for Rendang:")
    for place in recommendations:
        print(f"\n{place['name']}")
        print(f"Cuisine: {place['cuisine']}")
        print(f"Location: {place['location']}")
        print(f"Rating: {place['rating']}")
        print(f"Features: {', '.join(place['features'])}") 