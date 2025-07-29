import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics.pairwise import cosine_similarity
import pickle


class ImageSearchSetup:
    def __init__(self, data_dir='data', output_dir='outputs'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def extract_features(self, image_path):
        """Extract features from a single image."""
        try:
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            features = self.model.predict(image, verbose=0)
            return features.flatten()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def get_image_paths(self):
        """Get all image file paths from the data directory."""
        print("Absolute path to data directory:", os.path.abspath(self.data_dir))

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"The specified directory '{self.data_dir}' does not exist!")

        # Common image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = []

        for fname in os.listdir(self.data_dir):
            full_path = os.path.join(self.data_dir, fname)
            if os.path.isfile(full_path) and any(fname.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(full_path)

        if len(image_paths) == 0:
            raise ValueError(f"No images found in the directory '{self.data_dir}'")

        print(f"Found {len(image_paths)} images")
        return image_paths

    def extract_all_features(self):
        """Extract features from all images and save results."""
        image_paths = self.get_image_paths()

        print("Extracting features from all images...")
        features_list = []
        valid_paths = []

        for i, path in enumerate(image_paths):
            if i % 100 == 0:
                print(f"Processing image {i + 1}/{len(image_paths)}")

            features = self.extract_features(path)
            if features is not None:
                features_list.append(features)
                valid_paths.append(path)

        features_array = np.array(features_list)

        # Save features and image paths
        np.save(os.path.join(self.output_dir, 'features.npy'), features_array)
        np.save(os.path.join(self.output_dir, 'image_paths.npy'), valid_paths)

        print(f"Features extracted and saved for {len(valid_paths)} images")
        print(f"Features shape: {features_array.shape}")

        return features_array, valid_paths

    def compute_similarity_matrix(self, features):
        """Compute and save similarity matrix."""
        print("Computing similarity matrix...")
        similarity_matrix = cosine_similarity(features)
        np.save(os.path.join(self.output_dir, 'similarity_matrix.npy'), similarity_matrix)
        print(f"Similarity matrix computed and saved. Shape: {similarity_matrix.shape}")
        return similarity_matrix

    def rank_normalization(self, similarity_matrix, L):
        """Apply rank normalization to similarity matrix."""
        print(f"Applying rank normalization with L={L}...")
        n = similarity_matrix.shape[0]
        normalized_matrix = np.zeros_like(similarity_matrix)

        for i in range(n):
            ranks = np.argsort(-similarity_matrix[i])[:L]
            for j in ranks:
                normalized_matrix[i, j] = similarity_matrix[i, j]

        np.save(os.path.join(self.output_dir, 'normalized_similarity_matrix.npy'), normalized_matrix)
        print("Rank normalized similarity matrix saved")
        return normalized_matrix

    def construct_hypergraph(self, normalized_matrix, k):
        """Construct hypergraph from normalized similarity matrix."""
        print(f"Constructing hypergraph with k={k}...")
        n = normalized_matrix.shape[0]
        hyperedges = []

        for i in range(n):
            neighbors = np.argsort(-normalized_matrix[i])[:k]
            hyperedges.append(neighbors)

        # Save hyperedges
        with open(os.path.join(self.output_dir, 'hyperedges.pkl'), 'wb') as f:
            pickle.dump(hyperedges, f)

        print("Hypergraph constructed and saved")
        return hyperedges

    def compute_hyperedge_similarity(self, hyperedges, n):
        """Compute hyperedge similarity matrix."""
        print("Computing hyperedge similarity...")
        incidence_matrix = np.zeros((n, n))

        for edge in hyperedges:
            for i in edge:
                for j in edge:
                    if i != j:
                        incidence_matrix[i, j] += 1

        np.save(os.path.join(self.output_dir, 'hyperedge_similarity.npy'), incidence_matrix)
        print("Hyperedge similarity matrix saved")
        return incidence_matrix

    def run_full_setup(self, L=50, k=10):
        """Run the complete setup pipeline."""
        print("Starting full setup pipeline...")

        # Extract features
        features, image_paths = self.extract_all_features()

        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(features)

        # Apply rank normalization
        normalized_matrix = self.rank_normalization(similarity_matrix, L)

        # Construct hypergraph
        hyperedges = self.construct_hypergraph(normalized_matrix, k)

        # Compute hyperedge similarity
        hyperedge_similarity = self.compute_hyperedge_similarity(hyperedges, len(image_paths))

        print("Setup complete! All files saved to outputs/ directory:")
        print("- features.npy")
        print("- image_paths.npy")
        print("- similarity_matrix.npy")
        print("- normalized_similarity_matrix.npy")
        print("- hyperedges.pkl")
        print("- hyperedge_similarity.npy")


if __name__ == "__main__":
    # Update this path to your data directory
    data_directory = r'C:\Users\nikos\PycharmProjects\imageAnalysis\data'

    setup = ImageSearchSetup(data_dir=data_directory)
    setup.run_full_setup(L=50, k=10)