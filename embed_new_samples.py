import torch
import pandas as pd
import numpy as np
from pathlib import Path


def load_model(model_path, device='cpu'):
    """
    Load the pretrained MAE-30% gut metagenomics encoder model.

    Args:
        model_path: Path to the mae30_encoder_full.pkl file
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        model: Pretrained encoder ready for inference
    """
    model = torch.load(model_path, map_location=device)
    model.eval()
    model.to(device)
    return model


def load_species_reference(reference_path):
    """
    Load the species reference list that defines the model's expected input order.

    Args:
        reference_path: Path to the model_species_reference.csv file

    Returns:
        list: Ordered list of species names expected by the model
    """
    species_ref = pd.read_csv(reference_path)
    return species_ref['species_name'].tolist()


def arrange_input(input_samples_df, species_reference_path):
    """
    Preprocess and align input samples with the model's expected format.

    Steps:
    1. Reorder species according to model's expected order
    2. Remove species not in the model's reference
    3. Add missing species with 0 values
    4. Replace 0 values with 0.0001 (detection threshold)
    5. Apply log10 transformation
    6. Shift values by +2

    Args:
        input_samples_df: DataFrame with samples as rows, species as columns
        species_reference_path: Path to model_species_reference.csv

    Returns:
        tuple: (torch.Tensor, pd.DataFrame) - preprocessed tensor and DataFrame
    """
    # Load expected species order
    expected_species = load_species_reference(species_reference_path)

    print(f"Expected species count: {len(expected_species)}")
    print(f"Input species count: {input_samples_df.shape[1]}")
    print(f"Input samples count: {input_samples_df.shape[0]}")

    # Check for overlap
    input_species = set(input_samples_df.columns)
    expected_species_set = set(expected_species)

    overlap = input_species.intersection(expected_species_set)
    missing_in_input = expected_species_set - input_species
    extra_in_input = input_species - expected_species_set

    print(f"\nSpecies alignment summary:")
    print(f"  - Species overlap: {len(overlap)}/{len(expected_species)} ({100 * len(overlap) / len(expected_species):.1f}%)")
    print(f"  - Missing from input: {len(missing_in_input)}")
    print(f"  - Extra in input (will be ignored): {len(extra_in_input)}")

    if len(overlap) < len(expected_species) * 0.5:  # Less than 50% overlap
        print("WARNING: Low species overlap. Results may be unreliable.")

    # Create aligned DataFrame
    aligned_df = pd.DataFrame(index=input_samples_df.index, columns=expected_species)

    # Fill with input data where available
    # Handle species mapping with abundance splitting for duplicates
    print("\nMapping species and handling duplicates...")
    for species in expected_species:
        if species in input_samples_df.columns:
            # Count how many times this species appears in the expected list
            species_count = expected_species.count(species)
            if species_count > 1:
                print(f"  - Species '{species}' appears {species_count} times, splitting abundance equally")

            # Split the abundance equally among duplicate entries
            aligned_df[species] = input_samples_df[species] / species_count
        else:
            aligned_df[species] = 0.0

    print(f"\nAlignment complete. Shape: {aligned_df.shape}")

    # Step 1: Replace 0 values with detection threshold (0.0001)
    processed_df = aligned_df.replace(to_replace=0.0, value=0.0001)

    # Step 2: Apply log10 transformation
    processed_df = np.log10(processed_df)

    # Step 3: Shift by +2 (converts range from [-4, 0] to [-2, 2])
    processed_df = processed_df + 2

    print(f"Preprocessing complete.")
    print(f"  - Value range: [{processed_df.values.min():.3f}, {processed_df.values.max():.3f}]")
    print(f"  - Mean: {processed_df.values.mean():.3f}, Std: {processed_df.values.std():.3f}")

    # Convert to tensor (consistent with training pipeline)
    preprocessed_tensor = torch.tensor(processed_df.values, dtype=torch.float32)

    return preprocessed_tensor, processed_df


def embed_samples(preprocessed_samples, preprocessed_df, model, device='cpu'):
    """
    Generate embeddings for preprocessed samples using the trained model.

    Args:
        preprocessed_samples: Tensor of preprocessed samples
        preprocessed_df: Original DataFrame (for getting the index)
        model: Loaded model
        device: Device to run inference on

    Returns:
        pd.DataFrame: Sample embeddings with sample IDs as index
    """

    model.eval()

    print(f"Generating embeddings for {len(preprocessed_samples)} samples...")

    # Generate representations in single batch
    with torch.no_grad():
        preprocessed_samples = preprocessed_samples.to(device)
        representations = model(preprocessed_samples)
        representations = representations.cpu().numpy()

    print(f"Embeddings generated. Shape: {representations.shape}")

    # Create DataFrame with sample IDs from the preprocessed DataFrame
    embeddings_df = pd.DataFrame(
        representations,
        index=preprocessed_df.index,  # Use index from preprocessed data
        columns=[f'embedding_{i}' for i in range(representations.shape[1])]
    )

    return embeddings_df


def save_output(embeddings_df, output_path):
    """
    Save sample embeddings DataFrame to CSV file.

    Args:
        embeddings_df: DataFrame with embeddings and sample IDs
        output_path: Path to save the embeddings (CSV format)
    """
    embeddings_df.to_csv(output_path)
    print(f"Embeddings saved to {output_path}")


def get_sample_representations(input_samples_df, output_path, model_path,
                               species_reference_path, device='cpu'):
    """
    Complete pipeline to generate sample embeddings from microbiome data.

    Args:
        input_samples_df: DataFrame with samples as rows, species as columns
        output_path: Path to save the embeddings (CSV format)
        model_path: Path to the pretrained model file
        species_reference_path: Path to model_species_reference.csv
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        pd.DataFrame: Generated sample representations DataFrame
    """

    # Step 1: Load model
    model = load_model(model_path, device)

    # Step 2: Preprocess input data
    preprocessed_tensor, preprocessed_df = arrange_input(input_samples_df, species_reference_path)

    # Step 3: Generate embeddings
    sample_representations = embed_samples(preprocessed_tensor, preprocessed_df, model, device)

    # Step 4: Save results
    save_output(sample_representations, output_path)

    return sample_representations


if __name__ == "__main__":
    # Example parameters - update these for your use case
    MODEL_FILES_DIR = Path("your_dir_path")
    MODEL_PATH = MODEL_FILES_DIR.joinpath("mae30_encoder_full.pkl")
    SPECIES_REFERENCE_PATH = MODEL_FILES_DIR.joinpath("model_species_reference.csv")
    OUTPUT_PATH = "sample_embeddings.csv"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load your microbiome data
    # input_df = pd.read_csv("your_microbiome_data.csv", index_col=0)

    # Run the pipeline
    # embeddings = get_sample_representations(
    #     input_samples_df=input_df,
    #     output_path=OUTPUT_PATH,
    #     model_path=MODEL_PATH,
    #     species_reference_path=SPECIES_REFERENCE_PATH,
    #     device=DEVICE
    # )

    print("Script ready! Update the paths and uncomment the example usage to run.")




