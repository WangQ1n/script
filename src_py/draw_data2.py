import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def plot_results(data):
    # Plot acceleration
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(data['timestamp'], data['acc_x'], label='Acc X')
    plt.plot(data['timestamp'], data['acc_y'], label='Acc Y')
    plt.plot(data['timestamp'], data['acc_z'], label='Acc Z')
    plt.title("Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.legend()
    plt.grid()
    
    # Plot angular velocity
    plt.subplot(2, 1, 2)
    plt.plot(data['timestamp'], data['ang_vel_x'], label='Ang Vel X')
    plt.plot(data['timestamp'], data['ang_vel_y'], label='Ang Vel Y')
    plt.plot(data['timestamp'], data['ang_vel_z'], label='Ang Vel Z')
    plt.title("Angular Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()



def calculate_differences(data):
    # Extract columns as numpy arrays for calculations
    timestamps = data['timestamp'].to_numpy()
    positions = data[['pos_x', 'pos_y', 'pos_z']].to_numpy()
    
    # Calculate time differences
    print(timestamps)
    dt = np.diff(timestamps, prepend=timestamps[0])
    dt[dt == 0] = 1e-9  # Avoid division by zero

    # Calculate velocities (dx/dt)
    velocity = np.diff(positions, axis=0, prepend=[positions[0]]) / dt[:, None]
    
    # Calculate accelerations (d²x/dt²)
    acceleration = np.diff(velocity, axis=0, prepend=[velocity[0]]) / dt[:, None]
    
    # Calculate angular velocities
    quaternions = data[['quat_w', 'quat_x', 'quat_y', 'quat_z']].to_numpy()
    rotations = R.from_quat(quaternions[:, [1, 2, 3, 0]])  # scipy expects (x, y, z, w)
    angular_velocity = rotations[1:].inv() * rotations[:-1]
    angular_velocity = angular_velocity.as_rotvec() / dt[1:, None]
    angular_velocity = np.vstack([angular_velocity[0], angular_velocity])  # Match length
    angular_velocity = angular_velocity * (180. / np.pi)
    
    return acceleration, angular_velocity

def main():
    # Load data from TXT
    txt_file = '/home/crrcdt123/git/fastlio2_ros2/map_innov_1226/keyframe.txt'
    
    # Specify column names explicitly
    columns = ['id', 'timestamp', 'pos_x', 'pos_y', 'pos_z', 'quat_w', 'quat_x', 'quat_y', 'quat_z']
    
    # Read TXT file without headers, assuming whitespace-separated values
    data = pd.read_csv(txt_file, delim_whitespace=True, header=None, names=columns, usecols=range(9))
    
    # Calculate acceleration and angular  velocity
    acceleration, angular_velocity = calculate_differences(data)
    
    # Add results to DataFrame
    data['acc_x'], data['acc_y'], data['acc_z'] = acceleration[:, 0], acceleration[:, 1], acceleration[:, 2]
    data['ang_vel_x'], data['ang_vel_y'], data['ang_vel_z'] = angular_velocity[:, 0], angular_velocity[:, 1], angular_velocity[:, 2]
    
    # Display as table
    print(data[['timestamp', 'pos_x', 'pos_y', 'pos_z', 'acc_x', 'acc_y', 'acc_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z']])
    
    # Save to CSV
    output_file = '/home/crrcdt123/git/fastlio2_ros2/output.csv'
    data.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    data.plot(x='timestamp', y=['acc_x', 'acc_y', 'acc_z'], title='Linear Acceleration')
    data.plot(x='timestamp', y=['ang_vel_x', 'ang_vel_y', 'ang_vel_z'], title='Angular Velocity')
    plt.show()
    # plot_results(data.to_numpy())

if __name__ == "__main__":
    main()
