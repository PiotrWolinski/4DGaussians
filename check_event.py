import torch

# Replace with the actual path to your events.pt file
events_file_path = "/home/debmalya/Documents/4DGaussians_events/4DGaussians/data/final_dataset/falling_bag/events_cam15.pt"

# Load the events.pt file
events_data = torch.load(events_file_path)

# Print the shape and a sample of the data
print(f"Shape of events.pt: {events_data.shape}")
print(f"Sample values from events.pt: {events_data[:30]}")  