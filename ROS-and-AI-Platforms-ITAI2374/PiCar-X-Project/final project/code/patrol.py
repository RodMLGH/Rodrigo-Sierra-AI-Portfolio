from picarx import Picarx
from time import sleep

# Initialize the Robot
px = Picarx()

# --- CONFIGURATION ---
POWER = 20                 # Motor speed (0-100) - Keep low for safety
SAFE_DISTANCE = 15         # centimeters
GM_THRESHOLD = 500         # Grayscale threshold (Adjust based on lighting)

def get_grayscale_status():
    """
    Reads the 3 grayscale sensors to determine where the line is.
    Returns: 'left', 'right', 'straight', or 'lost'
    """
    gm_val_list = px.get_grayscale_data()
    # gm_val_list looks like [left_sensor, center_sensor, right_sensor]

    # Logic: If sensor value is LOW, it sees Black (Line). High = White (Floor).
    # Note: Adjust logic depending on if your line is black or white.
    # Assuming Black Line on White Floor:
    left = gm_val_list[0] < GM_THRESHOLD
    center = gm_val_list[1] < GM_THRESHOLD
    right = gm_val_list[2] < GM_THRESHOLD

    if center:
        return 'straight'
    elif left:
        return 'left'
    elif right:
        return 'right'
    else:
        return 'lost'

def safety_check():
    # Placeholder for checking the ultrasonic sensor distance
    # The image doesn't show the implementation, but this is its purpose.
    distance = px.ultrasonic_distance()
    return distance > SAFE_DISTANCE # Returns True if safe

def main():
    print("🚜 Autonomous Patrol Started")
    print(" (Press Ctrl+C to stop manually.)")

    try:
        while True:
            # 1. PERCEPTION & SAFETY (Ethical Operation)
            if not safety_check():
                px.stop()
                print("⚠️ OBSTACLE DETECTED! Emergency Stop.")
                sleep(0.1)
                continue # Skip the rest of the loop, wait for path to clear

            # 2. NAVIGATION LOGIC (Real-time decision making)
            status = get_grayscale_status()

            if status == 'straight':
                px.set_dir_servo_angle(0)    # Wheels straight
                px.forward(POWER)
            elif status == 'left':
                px.set_dir_servo_angle(-35)  # Turn wheels left
                px.forward(POWER)
            elif status == 'right':
                px.set_dir_servo_angle(35)   # Turn wheels right
                px.forward(POWER)
            elif status == 'lost':
                # Safety behavior: If line is lost, stop to prevent running away
                px.stop()

            sleep(0.01) # Small delay for stability

    except KeyboardInterrupt:
        print("\n🛑 User Interrupt. Stopping.")
    finally:
        px.stop()

if __name__ == "__main__":
    main()