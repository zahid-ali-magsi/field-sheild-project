def get_treatment(disease):
    treatments = {
        # === Rice Diseases ===
        'Bacterial Leaf Blight': [
            "Use resistant varieties like IR20",
            "Apply Streptomycin sulfate or Copper fungicide",
            "Maintain proper field drainage"
        ],
        'Brown Spot': [
            "Apply Azoxystrobin or Tricyclazole",
            "Use balanced fertilization",
            "Remove infected plant debris"
        ],
        'Leaf Smut': [
            "Use clean seeds",
            "Apply Propiconazole",
            "Practice crop rotation"
        ],
        'Healthy Rice Leaf': [
            "Maintain regular monitoring",
            "Follow good agricultural practices",
            "Ensure proper nutrient balance"
        ],

        # === Wheat Diseases ===
        'Wheat Crown Root Rot': [
            "Improve soil drainage to avoid excess moisture",
            "Apply fungicides like Propiconazole or Carbendazim",
            "Rotate crops to reduce pathogen load",
            "Use resistant wheat varieties when possible"
        ],
        'Wheat Leaf Rust': [
            "Spray fungicides like Mancozeb or Propiconazole",
            "Avoid excessive nitrogen application",
            "Plant rust-resistant wheat varieties",
            "Monitor fields during humid weather for early detection"
        ],
        'Wheat Loose Smut': [
            "Use certified smut-free seeds",
            "Treat seeds with fungicides such as Carboxin or Tebuconazole",
            "Remove and destroy infected plants",
            "Follow crop rotation to reduce disease spread"
        ],
        'Wheat Healthy': [
            "Maintain good irrigation and drainage",
            "Use balanced fertilizer for strong growth",
            "Regular crop monitoring for early signs",
            "Adopt proper sanitation and crop rotation"
        ]
    }

    return treatments.get(disease, ["No specific treatment recommended. Consult an expert."])




# def get_treatment(disease):
#     treatments = {
#         'Bacterial Leaf Blight': [
#             "Use resistant varieties like IR20",
#             "Apply Streptomycin sulfate or Copper fungicide",
#             "Maintain proper field drainage"
#         ],
#         'Brown Spot': [
#             "Apply Azoxystrobin or Tricyclazole",
#             "Use balanced fertilization",
#             "Remove infected plant debris"
#         ],
#         'Leaf Smut': [
#             "Use clean seeds",
#             "Apply Propiconazole",
#             "Practice crop rotation"
#         ],
#         'Healthy Rice Leaf': [
#             "Maintain regular monitoring",
#             "Follow good agricultural practices",
#             "Ensure proper nutrient balance"
#         ]
#     }
#     return treatments.get(disease, ["No specific treatment recommended"])

