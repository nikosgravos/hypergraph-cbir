import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from search_utils import ImageSearchUtils


def display_search_results(query_image_path, search_utils, k=10):
    """Display search results in a grid layout."""

    # Check if query image exists
    if not os.path.exists(query_image_path):
        print(f"Error: Query image '{query_image_path}' not found!")
        return

    print(f"Searching for images similar to: {query_image_path}")

    try:
        # Get top-k similar images
        top_k_images = search_utils.retrieve_top_k_similar(query_image_path, k=k)

        # Calculate grid layout
        total_images = len(top_k_images) + 1  # +1 for query image
        images_per_row = 5
        n_rows = -(-total_images // images_per_row)  # Ceiling division

        # Create figure
        fig, axes = plt.subplots(n_rows, images_per_row,
                                 figsize=(images_per_row * 4, n_rows * 4))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        # Display query image
        try:
            query_image = Image.open(query_image_path)
            axes[0].imshow(query_image)
            query_filename = query_image_path.split("/")[-1].split("\\")[-1]
            axes[0].set_title(f"Query Image\n{query_filename}",
                              fontsize=14, fontweight="bold")
            axes[0].axis("off")
        except Exception as e:
            print(f"Error loading query image: {e}")
            return

        # Display similar images
        for i, (path, similarity) in enumerate(top_k_images):
            try:
                image = Image.open(path)
                file_name = path.split("/")[-1].split("\\")[-1]
                axes[i + 1].imshow(image)
                axes[i + 1].set_title(f"Rank {i + 1}\n{file_name}\nSim: {similarity:.4f}",
                                      fontsize=10)
                axes[i + 1].axis("off")
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                axes[i + 1].set_title(f"Error loading\nRank {i + 1}", fontsize=10)
                axes[i + 1].axis("off")

        # Hide unused subplots
        for i in range(len(top_k_images) + 1, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

        # Print text results
        print(f"\nSearch Results for: {query_filename}")
        print("-" * 60)
        for i, (path, similarity) in enumerate(top_k_images):
            file_name = path.split("/")[-1].split("\\")[-1]
            print(f'Rank {i + 1:2d}: {file_name:<25} | Similarity: {similarity:.4f}')

    except Exception as e:
        print(f"Error during search: {e}")


def interactive_search(search_utils):
    """Interactive search mode."""
    print("\n" + "=" * 60)
    print("INTERACTIVE IMAGE SEARCH MODE")
    print("=" * 60)
    print("Commands:")
    print("- Enter image path to search")
    print("- 'stats' to show dataset statistics")
    print("- 'error' to calculate ranking errors")
    print("- 'list' to show some available images")
    print("- 'quit' to exit")
    print("-" * 60)

    while True:
        user_input = input("\nEnter command or image path: ").strip()

        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        elif user_input.lower() == 'stats':
            search_utils.get_dataset_stats()
        elif user_input.lower() == 'error':
            num_queries = input("Enter number of queries for error calculation (default 100): ").strip()
            try:
                num_queries = int(num_queries) if num_queries else 100
                search_utils.calculate_ranking_errors(num_queries=num_queries, k=10)
            except ValueError:
                print("Invalid number. Using default (100).")
                search_utils.calculate_ranking_errors(num_queries=100, k=10)
        elif user_input.lower() == 'list':
            print("Sample available images:")
            for i, path in enumerate(search_utils.image_paths[:20]):
                file_name = path.split("/")[-1].split("\\")[-1]
                print(f"  {file_name}")
            if len(search_utils.image_paths) > 20:
                print(f"  ... and {len(search_utils.image_paths) - 20} more images")
        elif user_input:
            if os.path.exists(user_input):
                k = input("Enter number of results to show (default 10): ").strip()
                try:
                    k = int(k) if k else 10
                    display_search_results(user_input, search_utils, k=k)
                except ValueError:
                    print("Invalid number. Using default (10).")
                    display_search_results(user_input, search_utils, k=10)
            else:
                print(f"File not found: {user_input}")


def main():
    """Main function."""
    print("Image Similarity Search System")
    print("=" * 50)

    # Check if required files exist
    required_files = ['outputs/features.npy', 'outputs/image_paths.npy']
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print("Error: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease run setup.py first to extract features and prepare data.")
        return

    try:
        # Initialize search utilities
        search_utils = ImageSearchUtils()

        # Example usage with a specific image (update this path)
        example_query = 'data/34019.jpeg'

        if os.path.exists(example_query):
            print(f"\nRunning example search with: {example_query}")
            display_search_results(example_query, search_utils, k=10)
        else:
            print(f"Example query image '{example_query}' not found.")
            print("Available images:")
            for i, path in enumerate(search_utils.image_paths[:10]):
                file_name = path.split("/")[-1].split("\\")[-1]
                print(f"  {file_name}")
            if len(search_utils.image_paths) > 10:
                print(f"  ... and {len(search_utils.image_paths) - 10} more")

        # Start interactive mode
        interactive_search(search_utils)

    except Exception as e:
        print(f"Error initializing search system: {e}")
        print("Make sure you have run setup.py and have the required files.")


if __name__ == "__main__":
    main()