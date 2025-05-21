import tkinter as tk
from virtual_camera_simulator import VirtualCameraSimulator

# --- Main Execution ---
if __name__ == "__main__":
    main_root = tk.Tk()
    app = VirtualCameraSimulator(main_root)
    main_root.mainloop()
