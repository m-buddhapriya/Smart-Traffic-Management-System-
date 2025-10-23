import os

def distribute_green_time(total_cycle_time, vehicle_counts):
    total_vehicles = sum(vehicle_counts.values())
    green_times = {}

    for direction, count in vehicle_counts.items():
        time = (count / total_vehicles) * total_cycle_time if total_vehicles > 0 else total_cycle_time / 4
        green_times[direction] = round(time)

    return green_times

def main():
    base_dir = r"D:\Project Btech\Dynamic Signal"
    total_cycle_time = 130  # Total signal cycle time

    # Static/default values for other directions (tweak these if needed)
    vehicle_counts = {
        'North': 5,
        'South': 8,
        'West': 7
    }

    # Load the actual detected vehicle count for East direction
    east_file = os.path.join(base_dir, "east_vehicle_count.txt")
    try:
        with open(east_file, "r") as file:
            content = file.read().strip()
            if content.isdigit():
                vehicle_counts['East'] = int(content)
            else:
                print("Invalid value in east_vehicle_count.txt. Using default (10).")
                vehicle_counts['East'] = 10
    except FileNotFoundError:
        print("east_vehicle_count.txt not found. Using default (10).")
        vehicle_counts['East'] = 10

    # Distribute green time
    green_times = distribute_green_time(total_cycle_time, vehicle_counts)

    print("\nAdjusted Green Signal Timings:")
    for direction, time in green_times.items():
        print(f"{direction}: {time} seconds")

if __name__ == "__main__":
    main()
