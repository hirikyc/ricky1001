import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from foodlens_with_recommendation import FoodLensWithRecommendation

# Try importing OpenCV, if not available, show error message
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    st.error("""
    OpenCV is not installed. Please install it using:
    ```
    pip install opencv-python
    ```
    or if using Anaconda:
    ```
    conda install opencv
    ```
    """)

# Try importing the FoodLens system
try:
    from foodlens_with_recommendation import FoodLensWithRecommendation
    FOODLENS_AVAILABLE = True
except ImportError as e:
    FOODLENS_AVAILABLE = False
    st.error(f"""
    Error importing FoodLens system: {str(e)}
    Please make sure all required files are in the correct directory:
    - foodlens_with_recommendation.py
    - food_recommendation.py
    - model_indonesian_food.keras
    - places_data.json
    """)

# Set page configuration
st.set_page_config(
    page_title="FoodLens - Pengenalan Makanan & Rekomendasi Restoran",
    page_icon="🍽️",
    layout="wide"
)

# Mapping from food name to cuisine (asal makanan)
FOOD_TO_CUISINE = {
    "bakso": "Jawa",
    "ayam goreng": "Nasional",
    "gulai tunjang": "Sumatera Barat",
    "telur dadar": "Nasional",
    "ayam pop": "Sumatera Barat",
    "telur balado": "Sumatera Barat",
    "gulai ikan": "Sumatera Barat",
    "dendeng batokok": "Sumatera Barat",
    "daging rendang": "Sumatera Barat",
    "gulai tambusu": "Sumatera Barat",
    "kue cubit": "Jakarta",
    "seblak": "Jawa Barat",
    "risol": "Nasional",
    "pepes ikan": "Jawa Barat",
    "bika ambon": "Sumatera Utara",
    "mie ayam": "Solo /Yogyakarta",
    "kue lapis": "Nasional",
    "dadar gulung": "Jawa",
    "klepon": "Jawa Timur / Jawa Tengah",
    "ayam geprek": "Yogyakarta",
    "tempe bacem": "Jawa Tengah / Yogyakarta",
    "ketoprak": "Jakarta",
    "bakwan": "Jawa Timur",
    "gulai kambing": "Sumatera Barat",
    "opor ayam": "Jawa Tengah / Yogyakarta",
    "putu ayu": "Jawa",
    "martabak manis": "Bengkulu",
    "putri salju": "Jawa",
    "serabi": "Jawa Barat",
    "bubur ayam": "Jakarta / Jawa Barat",
    "pempek": "Palembang",
    "bebek betutu": "Bali",
    "gudeg": "Yogyakarta",
    "papeda": "Papua",
    "tape": "Jawa Barat / Jawa Tengah",
    "ayam bakar": "Nasional",
    "mie goreng": "Nasional",
    "rawon": "Jawa Timur",
    "nasi goreng": "Nasional",
    "sate": "Madura",
    "gado gado": "Jakarta",
    "ikan goreng": "Nasional",
    "coto makassar": "Makassar",
    "Soto": "Lamongan",
    "cireng": "Jawa Barat",
    "tahu sumedang": "Jawa Barat",
    "kerak telor": "Jakarta",
    "mie aceh": "Aceh",
    "beberuk terong": "Nusa Tenggara Barat",
    "nasi kuning": "Jawa",
}

# Mapping kota dan kecamatan untuk pencocokan yang lebih fleksibel
CITY_MAPPING = {
    "jakarta": ["jakarta", "jakarta pusat", "jakarta selatan", "jakarta timur", "jakarta barat", "jakarta utara"],
    "bandung": ["bandung", "bandung kota", "bandung barat"],
    "surabaya": ["surabaya", "surabaya kota", "surabaya timur", "surabaya barat"],
    "yogyakarta": ["yogyakarta", "jogja", "yogyakarta kota"],
    "semarang": ["semarang", "semarang kota"],
    "medan": ["medan", "medan kota"],
    "makassar": ["makassar", "makassar kota"],
    "palembang": ["palembang", "palembang kota"],
    "denpasar": ["denpasar", "denpasar kota"],
    "malang": ["malang", "malang kota"],
    "bogor": ["bogor", "bogor kota"],
    "solo": ["solo", "surakarta", "solo kota"],
    "padang": ["padang", "padang kota"],
    "pekanbaru": ["pekanbaru", "pekanbaru kota"],
    "balikpapan": ["balikpapan", "balikpapan kota"],
    "manado": ["manado", "manado kota"],
    "bali": ["bali", "denpasar", "kuta", "ubud", "seminyak"],
    "aceh": ["aceh", "banda aceh", "aceh besar"],
    "papua": ["papua", "jayapura", "papua barat"],
    "ntb": ["ntb", "lombok", "mataram", "nusa tenggara barat"],
    "ntt": ["ntt", "kupang", "nusa tenggara timur"],
    "sulawesi": ["sulawesi", "makassar", "manado", "palu"],
    "sumatera": ["sumatera", "medan", "palembang", "padang", "pekanbaru"],
    "jawa": ["jawa", "jakarta", "bandung", "surabaya", "yogyakarta", "semarang", "malang", "bogor", "solo"],
    "kalimantan": ["kalimantan", "balikpapan", "samarinda", "pontianak", "banjarmasin"],
}

# Price range mapping
PRICE_RANGES = {
    "murah": ["murah", "ekonomis", "warung", "rumah makan"],
    "sedang": ["sedang", "cafe", "restoran"],
    "mahal": ["mahal", "fine dining", "restaurant"]
}

def get_matching_cities(input_city):
    """Get all possible matching cities based on input"""
    input_city = input_city.lower()
    matching_cities = []
    
    # Check direct mapping
    if input_city in CITY_MAPPING:
        matching_cities.extend(CITY_MAPPING[input_city])
    else:
        # Check if input is part of any city mapping
        for city, aliases in CITY_MAPPING.items():
            if input_city in aliases or any(input_city in alias for alias in aliases):
                matching_cities.extend(aliases)
    
    # If no matches found, return the original input
    return matching_cities if matching_cities else [input_city]

def get_price_range(place_name, features):
    """Determine price range based on place name and features"""
    text = f"{place_name} {' '.join(features)}".lower()
    
    if any(term in text for term in PRICE_RANGES["mahal"]):
        return "💰💰💰"
    elif any(term in text for term in PRICE_RANGES["sedang"]):
        return "💰💰"
    else:
        return "💰"

def sort_recommendations(recommendations, sort_by="rating"):
    """Sort recommendations based on criteria"""
    if sort_by == "rating":
        return sorted(recommendations, key=lambda x: x['rating'], reverse=True)
    elif sort_by == "similarity":
        return sorted(recommendations, key=lambda x: x['similarity_score'], reverse=True)
    return recommendations

# Initialize the FoodLens system only if all dependencies are available
@st.cache_resource
def load_foodlens():
    if not OPENCV_AVAILABLE or not FOODLENS_AVAILABLE:
        return None
    return FoodLensWithRecommendation()

foodlens = load_foodlens()

# Title and description
st.title("🍽️ FoodLens - Pengenalan Makanan & Rekomendasi Restoran")
st.markdown("""
Unggah gambar makanan untuk:
1. Mengenali makanan dalam gambar
2. Mendapatkan rekomendasi restoran yang menyajikan makanan tersebut
""")

# Only proceed if all dependencies are available
if OPENCV_AVAILABLE and FOODLENS_AVAILABLE:
    # City input
    city = st.text_input("Masukkan kota Anda (contoh: Jakarta, Bandung):", "Jakarta")
    
    # Add sorting options
    sort_by = st.selectbox(
        "Urutkan rekomendasi berdasarkan:",
        ["Rating", "Tingkat Kecocokan"],
        index=0
    ).lower().replace(" ", "_")
    
    # File uploader
    uploaded_file = st.file_uploader("Pilih gambar makanan...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang Diunggah", use_column_width=True)
            
            # Convert PIL Image to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Save temporarily
            temp_path = "temp_image.jpg"
            cv2.imwrite(temp_path, opencv_image)
            
            # Process the image
            with st.spinner("Menganalisis gambar dan mencari rekomendasi..."):
                results = foodlens.process_image_and_recommend(temp_path)
                
                # Display recognition results
                st.markdown("## 🍲 Hasil Pengenalan Makanan")
                
                # Create columns for the main prediction
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Makanan yang Dikenali",
                        results['recognition']['food_name']
                    )
                with col2:
                    st.metric(
                        "Tingkat Kepercayaan",
                        f"{results['recognition']['confidence']:.2%}"
                    )
                
                # Display all predictions in an expandable section
                with st.expander("Lihat Semua Prediksi"):
                    st.markdown("### Semua Prediksi")
                    all_predictions = results['recognition']['all_predictions']
                    
                    # Sort predictions by confidence
                    all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
                    
                    # Create a table of all predictions
                    prediction_data = {
                        "Nama Makanan": [p['food_name'] for p in all_predictions],
                        "Tingkat Kepercayaan": [f"{p['confidence']:.2%}" for p in all_predictions]
                    }
                    st.table(prediction_data)
                
                # Display top 3 predictions
                st.markdown("### 3 Prediksi Teratas")
                top_predictions = results['recognition']['top_predictions']
                
                # Create a progress bar for each prediction
                for pred in top_predictions:
                    st.progress(pred['confidence'])
                    st.markdown(f"**{pred['food_name']}**: {pred['confidence']:.2%}")
                
                # Display recommendations
                st.markdown("## 🏪 Tempat yang Direkomendasikan")
                
                if results['recommendations']:
                    recognized_food = results['recognition']['food_name'].lower()
                    asal_makanan = FOOD_TO_CUISINE.get(recognized_food, None)
                    
                    # Filter recommendations by city/location (more flexible matching)
                    filtered_recommendations = [
                        place for place in results['recommendations']
                        if 'location' in place and city.lower() in place['location'].lower()
                    ]
                    # Sort recommendations
                    filtered_recommendations = sort_recommendations(filtered_recommendations, sort_by)

                    if filtered_recommendations:
                        st.success(f"Ditemukan {len(filtered_recommendations)} rekomendasi di {city} dan sekitarnya!")
                        # Group recommendations by cuisine
                        cuisine_groups = {}
                        for place in filtered_recommendations:
                            cuisine = place['cuisine']
                            if cuisine not in cuisine_groups:
                                cuisine_groups[cuisine] = []
                            cuisine_groups[cuisine].append(place)
                        # Display recommendations by cuisine
                        for cuisine, places in cuisine_groups.items():
                            st.markdown(f"### 🍽️ Restoran {cuisine}")
                            for i, place in enumerate(places, 1):
                                with st.expander(f"{i}. {place['name']} - Rating: {place['rating']}⭐"):
                                    st.markdown(
                                        f"""
                                        <div style='line-height: 1.8; font-size: 1.1em'>
                                        <b>Lokasi Restoran:</b> {place['location']}  <br>
                                        <b>Asal Makanan:</b> {asal_makanan if asal_makanan else place['cuisine']}  <br>
                                        <b>Lokasi Anda:</b> {city}  <br>
                                        <b>Tingkat Kecocokan:</b> {place['similarity_score']:.2%}  <br>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                    else:
                        st.warning(f"Tidak ditemukan rekomendasi di {city}. Menampilkan rekomendasi dari kota lain:")
                        # Get all recommendations and group by city
                        all_recommendations = [
                            place for place in results['recommendations']
                            if 'location' in place
                        ]
                        # Sort recommendations
                        all_recommendations = sort_recommendations(all_recommendations, sort_by)
                        if all_recommendations:
                            # Group recommendations by city
                            city_groups = {}
                            for place in all_recommendations:
                                city_name = place['location'].split(',')[0].strip()  # Get main city name
                                if city_name not in city_groups:
                                    city_groups[city_name] = []
                                city_groups[city_name].append(place)
                            # Display recommendations by city
                            for city_name, places in city_groups.items():
                                st.markdown(f"### 🏙️ Rekomendasi di {city_name}")
                                # Group by cuisine within each city
                                cuisine_groups = {}
                                for place in places:
                                    cuisine = place['cuisine']
                                    if cuisine not in cuisine_groups:
                                        cuisine_groups[cuisine] = []
                                    cuisine_groups[cuisine].append(place)
                                # Display recommendations by cuisine
                                for cuisine, cuisine_places in cuisine_groups.items():
                                    st.markdown(f"#### 🍽️ Restoran {cuisine}")
                                    for i, place in enumerate(cuisine_places, 1):
                                        with st.expander(f"{i}. {place['name']} - Rating: {place['rating']}⭐"):
                                            st.markdown(
                                                f"""
                                                <div style='line-height: 1.8; font-size: 1.1em'>
                                                <b>Lokasi Restoran:</b> {place['location']}  <br>
                                                <b>Asal Makanan:</b> {asal_makanan if asal_makanan else place['cuisine']}  <br>
                                                <b>Lokasi Anda:</b> {city}  <br>
                                                <b>Tingkat Kecocokan:</b> {place['similarity_score']:.2%}  <br>
                                                </div>
                                                """,
                                                unsafe_allow_html=True
                                            )
                        else:
                            st.error("Tidak ditemukan rekomendasi restoran untuk makanan ini.")
                
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {str(e)}")
else:
    st.warning("Silakan instal semua dependensi yang diperlukan untuk menggunakan aplikasi.")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Dibuat dengan ❤️ oleh Tim FoodLens</p>
</div>
""", unsafe_allow_html=True) 