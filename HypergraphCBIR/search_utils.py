import numpy as np
import random
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img


class ImageSearchUtils:
    def __init__(self, features_path='outputs/features.npy',
                 image_paths_path='outputs/image_paths.npy'):
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.features = np.load(features_path)
        self.image_paths = np.load(image_paths_path, allow_pickle=True)

        print(f"Loaded {len(self.image_paths)} images")
        print(f"Features shape: {self.features.shape}")

    def extract_features(self, image_path):
        """Extract features from the input image."""
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        features = self.model.predict(image, verbose=0)
        return features.flatten()

    def retrieve_top_k_similar(self, query_image_path, k=5):
        """Retrieve top-k most similar images to the query."""
        query_features = self.extract_features(query_image_path)

        # Compute cosine similarities
        similarities = np.dot(self.features, query_features) / (
                np.linalg.norm(self.features, axis=1) * np.linalg.norm(query_features)
        )

        # Get top-k indices
        top_k_indices = np.argsort(-similarities)[:k]

        # Return list of (image_path, similarity) tuples
        top_k_images = [(self.image_paths[idx], similarities[idx]) for idx in top_k_indices]
        return top_k_images

    def calculate_ranking_errors(self, num_queries=1000, k=10, save_results=True):
        """Calculate mean squared error for each rank position."""
        print(f"Calculating ranking errors using {num_queries} random queries...")

        # Sample random query images
        random_query_images = random.sample(list(self.image_paths),
                                            min(num_queries, len(self.image_paths)))

        similarity_sums = np.zeros(k)
        error_sums = np.zeros(k)
        num_processed = 0

        # Iterate over query images
        for i, query_image_path in enumerate(random_query_images):
            if i % 100 == 0:
                print(f"Processing query {i + 1}/{len(random_query_images)}")

            try:
                top_k_images = self.retrieve_top_k_similar(query_image_path, k=k)

                for rank, (_, similarity) in enumerate(top_k_images):
                    similarity_sums[rank] += similarity
                    error_sums[rank] += (1 - similarity) ** 2

                num_processed += 1

            except Exception as e:
                print(f"Error processing {query_image_path}: {e}")
                continue

        if num_processed == 0:
            raise ValueError("No queries were successfully processed!")

        # Calculate averages
        average_similarities = similarity_sums / num_processed
        mse_per_rank = error_sums / num_processed

        # Save results if requested
        if save_results:
            results = {
                'average_similarities': average_similarities,
                'mse_per_rank': mse_per_rank,
                'num_queries_processed': num_processed,
                'k': k
            }
            np.save('outputs/ranking_errors.npy', results)
            print("Ranking error results saved to outputs/ranking_errors.npy")

        # Print results
        print(f"\nRanking Error Analysis (based on {num_processed} queries):")
        print("-" * 60)
        for rank, (avg_sim, mse) in enumerate(zip(average_similarities, mse_per_rank), start=1):
            print(f'Rank {rank:2d}: Avg Similarity = {avg_sim:.4f}, MSE = {mse:.4f}')

        return average_similarities, mse_per_rank

    def search_and_display_info(self, query_image_path, k=10):
        """Search for similar images and return detailed information."""
        print(f"Searching for images similar to: {query_image_path}")

        try:
            top_k_images = self.retrieve_top_k_similar(query_image_path, k=k)

            print(f"\nTop {k} similar images:")
            print("-" * 80)

            results = []
            for i, (path, similarity) in enumerate(top_k_images):
                file_name = path.split("/")[-1].split("\\")[-1]  # Handle both Unix and Windows paths
                print(f'Rank {i + 1:2d}: {file_name:<25} | Similarity: {similarity:.4f}')
                results.append({
                    'rank': i + 1,
                    'path': path,
                    'filename': file_name,
                    'similarity': similarity
                })

            return results

        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def get_dataset_stats(self):
        """Get basic statistics about the dataset."""
        stats = {
            'total_images': len(self.image_paths),
            'feature_dimension': self.features.shape[1],
            'feature_stats': {
                'mean': np.mean(self.features),
                'std': np.std(self.features),
                'min': np.min(self.features),
                'max': np.max(self.features)
            }
        }

        print("Dataset Statistics:")
        print(f"- Total images: {stats['total_images']}")
        print(f"- Feature dimension: {stats['feature_dimension']}")
        print(f"- Feature statistics:")
        print(f"  * Mean: {stats['feature_stats']['mean']:.4f}")
        print(f"  * Std: {stats['feature_stats']['std']:.4f}")
        print(f"  * Min: {stats['feature_stats']['min']:.4f}")
        print(f"  * Max: {stats['feature_stats']['max']:.4f}")

        return stats


if __name__ == "__main__":
    # Example usage
    search_utils = ImageSearchUtils()

    # Get dataset statistics
    search_utils.get_dataset_stats()

    # Calculate ranking errors
    search_utils.calculate_ranking_errors(num_queries=100, k=10)