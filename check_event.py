import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Replace with the actual path to your events.pt file
events_file_path = "/home/debmalya/Documents/4DGaussians_events/4DGaussians/data/temp_pt/events_cam15.pt"

# Load the events.pt file
events_data = torch.load(events_file_path)

# Check the shape of the tensor
print(f"Shape of events.pt: {events_data.shape}")  # Expected shape: [temporal, height, width]

# Initialize the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.25)  # Adjust space for the slider

# Display the first slice initially
temporal_dim = events_data.shape[0]
current_slice = events_data[0].cpu().numpy()
im = ax.imshow(current_slice, cmap="viridis", interpolation="nearest")
plt.colorbar(im, ax=ax, label="Event Intensity")
ax.set_title("Event Data Slice at Timestamp 0")
ax.set_xlabel("Width")
ax.set_ylabel("Height")

# Add a slider for selecting the timestamp
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])  # Position of the slider
slider = Slider(ax_slider, "Timestamp", 0, temporal_dim - 1, valinit=0, valstep=1)

# Update function for the slider
def update(val):
    timestamp = int(slider.val)
    current_slice = events_data[timestamp].cpu().numpy()
    im.set_data(current_slice)
    ax.set_title(f"Event Data Slice at Timestamp {timestamp}")
    fig.canvas.draw_idle()

# Connect the slider to the update function
slider.on_changed(update)

plt.show()