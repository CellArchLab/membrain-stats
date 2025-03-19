from matplotlib import pyplot as plt
import numpy as np


def collect_between_angles(
    nn_stats,
    all_nn1_distances,
    all_nn2_distances,
    all_nn3_distances,
    all_nn1_angles,
    all_nn2_angles,
    all_nn3_angles,
):

    distances1 = nn_stats["nn0_distance"].values
    distances2 = nn_stats["nn1_distance"].values
    distances3 = nn_stats["nn2_distance"].values

    between_angles1 = nn_stats["nn0_angle"].values
    between_angles2 = nn_stats["nn1_angle"].values
    between_angles3 = nn_stats["nn2_angle"].values

    all_nn1_distances.append(distances1)
    all_nn2_distances.append(distances2)
    all_nn3_distances.append(distances3)

    all_nn1_angles.append(between_angles1)
    all_nn2_angles.append(between_angles2)
    all_nn3_angles.append(between_angles3)

    return (
        all_nn1_distances,
        all_nn2_distances,
        all_nn3_distances,
        all_nn1_angles,
        all_nn2_angles,
        all_nn3_angles,
    )


def plot_orientations(
    all_nn1_distances,
    all_nn2_distances,
    all_nn3_distances,
    all_nn1_angles,
    all_nn2_angles,
    all_nn3_angles,
):
    # theoretical density function
    def density_function(y):
        return 3 * (180 - y) ** 2 / 180**3

    # aggregate all the data
    all_distances = np.concatenate(
        [
            np.expand_dims(np.concatenate(all_nn1_distances), axis=1),
            np.expand_dims(np.concatenate(all_nn2_distances), axis=1),
            np.expand_dims(np.concatenate(all_nn3_distances), axis=1),
        ],
        axis=1,
    )
    all_angles = np.concatenate(
        [
            np.expand_dims(np.concatenate(all_nn1_angles), axis=1),
            np.expand_dims(np.concatenate(all_nn2_angles), axis=1),
            np.expand_dims(np.concatenate(all_nn3_angles), axis=1),
        ],
        axis=1,
    )

    # remove NN distances that are too large
    max_distance = 600
    distance_mask = all_distances > max_distance
    all_angles[distance_mask] = 1e5

    min_angles = np.min(all_angles, axis=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    bins = np.linspace(0, 180, 30)
    plt.hist(
        min_angles,
        bins=bins,
        density=True,
        color="gray",
        label="minimum neighbor angle",
    )
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Density")
    plt.title("Minimum angle between neighbors")
    plt.plot(bins, density_function(bins), color="black", label="Theoretical density")
    plt.xlim(left=-1, right=181)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.tick_params(axis="both", labelsize=20)
    ax.set_ylim(0, 0.025)
    ax.set_yticks(np.arange(0, 0.025, 0.005))
    ax.set_xticks(np.arange(0, 181, 30))
    plt.legend(markerscale=2.5)
    plt.show()


def find_pos_index(pos, start_positions):
    for k, curpos in enumerate(start_positions):
        if np.linalg.norm(curpos - pos) < 1:
            return k

def compute_connection_angle(vec1, vec2):
    """
    Compute the angle between two vectors.
    - vec1: The vector formed by the last two elements of the chain.
    - vec2: The vector connecting the current ribosome to the candidate neighbor.
    """
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    return np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0))

def iterative_chain_building(positions, nearest_neighbors, between_angles, max_angle, max_angle_connection, max_distance):
    """
    Build chains iteratively, incorporating connection vectors dynamically.
    - positions: Array of ribosome positions (Nx3 array).
    - nearest_neighbors: List of nearest neighbor indices for each position (Nx3 array).
    - between_angles: List of angles between each position and its 3 nearest neighbors (Nx3 array).
    - max_angle: Maximum allowable angle in degrees for both between angles and connection angles.
    Returns:
    - List of chains (each chain is a list of position indices).
    """
    visited = set()
    chains = []
    max_angle_rad = np.radians(max_angle)  # Convert max angle to radians
    max_angle_connection_rad = np.radians(max_angle_connection)  # Convert max angle to radians
    
    for start in range(len(positions)):
        if start in visited:
            continue
        
        # Start a new chain
        chain = [start]
        visited.add(start)
        
        prev_vec = None  # No previous vector at the start
        current = start
        
        while True:
            # Get neighbors and their angles
            neighbors = nearest_neighbors[current]
            angles = between_angles[current]
            
            # Filter neighbors based on max_angle constraint
            valid_candidates = []
            for neighbor, angle in zip(neighbors, angles):
                if neighbor in visited:
                    continue

                if np.linalg.norm(positions[neighbor] - positions[current]) > max_distance:
                    continue
                
                # Check the "between angle" constraint
                angle_rad = np.radians(angle)
                if angle_rad > max_angle_rad:
                    continue
                
                # If there's a previous vector, check connection angle constraint
                if prev_vec is not None:
                    connection_vec = positions[neighbor] - positions[current]
                    connection_angle = compute_connection_angle(prev_vec, connection_vec)
                    if connection_angle > max_angle_connection_rad:
                        continue
                
                # Add valid candidate (neighbor, angle)
                valid_candidates.append((neighbor, angle_rad))
            
            # If no valid candidates remain, stop growing the chain
            if not valid_candidates:
                break
            
            # Choose the candidate with the smallest angle
            next_ribosome, _ = min(valid_candidates, key=lambda x: x[1])
            
            # Add the next ribosome to the chain
            chain.append(next_ribosome)
            visited.add(next_ribosome)
            
            # Update the previous vector and current ribosome
            if len(chain) > 1:
                prev_vec = positions[chain[-1]] - positions[chain[-2]]
            current = next_ribosome
        
        # Append the completed chain
        chains.append(chain)
    
    return chains


def merge_chains(chains, positions, nearest_neighbors, between_angles, max_distance, max_angle_merge_orientation, max_angle_merge_connection):
    """
    Merge chains if their end points are close enough and both orientation and connection vectors align.
    - chains: List of chains (each chain is a list of position indices).
    - positions: Array of ribosome positions (Nx3 array).
    - nearest_neighbors: Nearest neighbors for each position (list of lists of indices).
    - between_angles: Angles between each position and its nearest neighbors (list of lists of angles).
    - max_distance: Maximum distance between end points to consider merging.
    - max_angle_merge_orientation: Maximum angle in degrees for protein orientation alignment.
    - max_angle_merge_connection: Maximum angle in degrees for chain connection vector alignment.
    Returns:
    - Merged list of chains.
    """
    max_distance_sq = max_distance ** 2  # Square for distance comparison
    max_angle_merge_orientation_rad = np.radians(max_angle_merge_orientation)  # Convert to radians
    max_angle_merge_connection_rad = np.radians(max_angle_merge_connection)  # Convert to radians
    break_point = np.array([809.7629, 7.5467, 315.2689])
    merged = True
    while merged:
        merged = False
        break_flag = False
        for i in range(len(chains)):
            for j in range(i + 1, len(chains)):
                chain1, chain2 = chains[i], chains[j]
                if len(chain1) == 1 and len(chain2) == 1:
                    continue

                # Get endpoints of chains
                end1_idx, start2_idx = chain1[-1], chain2[0]
                start1_idx, end2_idx = chain1[0], chain2[-1]

                end1, start2 = positions[end1_idx], positions[start2_idx]
                start1, end2 = positions[start1_idx], positions[end2_idx]

                # Compute vectors
                vec_end1 = end1 - positions[chain1[-2]] if len(chain1) > 1 else None
                vec_start2 = start2 - positions[chain2[1]] if len(chain2) > 1 else None
                vec_end2 = end2 - positions[chain2[-2]] if len(chain2) > 1 else None
                vec_start1 = start1 - positions[chain1[1]] if len(chain1) > 1 else None

                # Check all 4 connection possibilities
                iterate_combos = [
                    (end1_idx, start2_idx, vec_end1, vec_start2),
                    (start1_idx, end2_idx, vec_start1, vec_end2),
                    (end1_idx, end2_idx, vec_end1, vec_end2),
                    (start1_idx, start2_idx, vec_start1, vec_start2)
                ]
                combo_tokens = [("end", "start"), ("start", "end"), ("end", "end"), ("start", "start")]
                scenarios = []
                for combo_nr, combo in enumerate(iterate_combos):
                    reverse_flag = combo_nr > 1
                    point1_idx, point2_idx, vec1, vec2 = combo
                    point1, point2 = positions[point1_idx], positions[point2_idx]
                    # print(point1, point2)
                    if np.sum((point1 - point2) ** 2) <= max_distance_sq:
                        
                        in_between_flag = False
                        # Check for an in-between angle
                        if point2_idx in nearest_neighbors[point1_idx]:
                            in_between_flag = True
                            in_between_angle = between_angles[point1_idx][nearest_neighbors[point1_idx].index(point2_idx)]
                        elif point1_idx in nearest_neighbors[point2_idx]:
                            in_between_flag = True
                            in_between_angle = between_angles[point2_idx][nearest_neighbors[point2_idx].index(point1_idx)]

                        # Validate in-between angle
                        if not in_between_flag or in_between_angle > max_angle_merge_orientation:
                            continue
                        
                        # Validate connection angles
                        angle_flag = True
                        if vec1 is not None:
                            angle_flag &= compute_connection_angle(vec1, point2 - point1) <= max_angle_merge_connection_rad
                            if np.linalg.norm(point1 - break_point) < 1 or np.linalg.norm(point2 - break_point) < 1:
                                print(compute_connection_angle(vec1, point2 - point1), 1, angle_flag)
                        if vec2 is not None:
                            angle_flag &= compute_connection_angle(vec2, point1 - point2) <= max_angle_merge_connection_rad
                            if np.linalg.norm(point1 - break_point) < 1 or np.linalg.norm(point2 - break_point) < 1:
                                print(compute_connection_angle(vec2, point1 - point2), 2, angle_flag)
                        if not angle_flag:
                            continue
                        
                        # Append the valid scenario
                        if reverse_flag:
                            scenarios.append((f"chain1_{combo_tokens[combo_nr][0]}_to_chain2_{combo_tokens[combo_nr][1]}", i, j))
                            # scenarios.append((f"chain1_{['end', 'start'][combo_nr % 2]}_to_chain2_{['start', 'end'][combo_nr % 2]}", i, j))
                        else:
                            scenarios.append((f"chain1_{combo_tokens[combo_nr][0]}_to_chain2_{combo_tokens[combo_nr][1]}", i, j))
                            # scenarios.append((f"chain1_{['end', 'start'][combo_nr % 2]}_to_chain2_{['end', 'start'][combo_nr % 2]}", i, j))
                        break
                # Handle the first valid scenario
                if scenarios:
                    scenario, chain1_idx, chain2_idx = scenarios[0]
                    if scenario == "chain1_end_to_chain2_start":
                        chains[chain1_idx].extend(chains[chain2_idx])
                    elif scenario == "chain1_start_to_chain2_end":
                        chains[chain2_idx].extend(chains[chain1_idx])
                        chains[chain1_idx] = chains[chain2_idx]
                    elif scenario == "chain1_end_to_chain2_end":
                        chains[chain1_idx].extend(reversed(chains[chain2_idx]))
                    elif scenario == "chain1_start_to_chain2_start":
                        chains[chain1_idx] = list(reversed(chains[chain1_idx])) + chains[chain2_idx]
                    # Remove the merged chain
                    chains.pop(chain2_idx)
                    merged = True
                    break
            if merged:# or break_flag:
                break  # Restart merging process after a merge

    return chains

colormaps = ['blue',
 'bop blue',
 'bop orange',
 'bop purple',
 'cyan',
 'green',
 'hsv',
 'I Blue',
 'I Bordeaux',
 'I Forest',
 'I Orange',
 'I Purple',
 'inferno',
 'magenta',
 'magma',
 'PiYG',
 'plasma',
 'red',
 'turbo',
 'twilight',
 'twilight_shifted',
 'viridis',
 'yellow']