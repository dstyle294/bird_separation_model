from datasets import load_dataset

birdset_data = load_dataset("DBD-research-group/BirdSet", "HSN")

print(birdset_data["test"]["audio"])