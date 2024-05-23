import os
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from scripts.ColorAnalyzer import ColorAnalyzer

# Configuration Constants
GOOD_CAKE_DIR = '/home/swordlord/crimson_tech/waiwai_cake/data/good_cake'
BAD_CAKE_DIR = '/home/swordlord/crimson_tech/waiwai_cake/data/bad_cake'
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def load_images_from_dir(dir_path):
    """Load image paths from a specified directory with given extensions."""
    return [os.path.join(dir_path, img) for img in os.listdir(dir_path) if os.path.splitext(img)[-1].lower() in IMAGE_EXTENSIONS]

def random_dir():
    """Randomly choose between the good and bad cake directories."""
    return random.choice([GOOD_CAKE_DIR, BAD_CAKE_DIR])

def analyze_image_pair(args):
    """Analyze the color difference between a good sample and a test sample image."""
    good_sample, sample = args
    color_analyzer = ColorAnalyzer(good_sample)
    try:
        result = color_analyzer.analyze(sample)
        mean_diff = np.mean(result["Color analysis result"])
        return sample, mean_diff
    except Exception as e:
        print(f"Error analyzing {sample} with {good_sample}: {e}")
        return sample, None

def main():
    results = defaultdict(list)
    
    good_samples = load_images_from_dir(GOOD_CAKE_DIR)
    
    for good_sample in good_samples:
        samples = load_images_from_dir(random_dir())
        pairs = [(good_sample, sample) for sample in samples]
        
        # Use multiprocessing to analyze pairs
        with Pool(cpu_count()) as pool:
            analysis_results = pool.map(analyze_image_pair, pairs)
        
        for sample, mean_diff in analysis_results:
            if mean_diff is not None:
                results[sample].append(mean_diff)
    
    # Aggregate results for analysis
    aggregated_results = {sample: np.mean(diffs) for sample, diffs in results.items() if diffs}
    
    # Analyze the results
    mean_diffs = np.array(list(aggregated_results.values()))
    
    print(f"Mean of differences: {np.mean(mean_diffs)}")
    print(f"Median of differences: {np.median(mean_diffs)}")
    print(f"Max of differences: {np.max(mean_diffs)}")
    print(f"Min of differences: {np.min(mean_diffs)}")
    print(f"Standard deviation of differences: {np.std(mean_diffs)}")
    print(f"Variance of differences: {np.var(mean_diffs)}")
    print(f"25th percentile of differences: {np.percentile(mean_diffs, 25)}")
    print(f"75th percentile of differences: {np.percentile(mean_diffs, 75)}")

if __name__ == '__main__':
    main()
