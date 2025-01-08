# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import subprocess as sp

import cv2

def get_resolution(filename):
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {filename}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height
            
def get_fps(filename):
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {filename}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def read_video(filename, skip=0, limit=-1):
    w, h = get_resolution(filename)
    
    command = ['ffmpeg',
            '-i', filename,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vsync', '0',
            '-vcodec', 'rawvideo', '-']
    
    i = 0
    with sp.Popen(command, stdout = sp.PIPE, bufsize=-1) as pipe:
        while True:
            data = pipe.stdout.read(w*h*3)
            if not data:
                break
            i += 1
            if i > limit and limit != -1:
                continue
            if i > skip:
                yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))
            
                
                
    
def downsample_tensor(X, factor):
    length = X.shape[0]//factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)

# ----------------------------------------------------------------------------------------------------------------------------
# Present 3d construction.

# def render_animation(keypoints, keypoints_metadata, poses, skeleton, fps, bitrate, azim, output, viewport,
#                      limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0, angles=None):
#     """
#     Render an animation. The supported output modes are:
#      -- 'interactive': display an interactive figure
#                        (also works on notebooks if associated with %matplotlib inline)
#      -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
#      -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
#      -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
#     """
#     import matplotlib.pyplot as plt
#     from matplotlib.animation import FuncAnimation, writers
#     import numpy as np

#     plt.ioff()
#     fig = plt.figure(figsize=(size * (1 + len(poses)), size))
#     ax_in = fig.add_subplot(1, 1 + len(poses), 1)
#     ax_in.get_xaxis().set_visible(False)
#     ax_in.get_yaxis().set_visible(False)
#     ax_in.set_axis_off()
#     ax_in.set_title('Input')

#     ax_3d = []
#     lines_3d = []
#     trajectories = []
#     radius = 1.7
#     for index, (title, data) in enumerate(poses.items()):
#         ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection='3d')
#         ax.view_init(elev=15., azim=azim)
#         ax.set_xlim3d([-radius / 2, radius / 2])
#         ax.set_zlim3d([0, radius])
#         ax.set_ylim3d([-radius / 2, radius / 2])
#         try:
#             ax.set_aspect('equal')
#         except NotImplementedError:
#             ax.set_aspect('auto')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_zticklabels([])
#         ax.dist = 7.5
#         ax.set_title(title)
#         ax_3d.append(ax)
#         lines_3d.append([])
#         trajectories.append(data[:, 0, [0, 1]])
#     poses = list(poses.values())

#     # Decode video
#     if input_video_path is None:
#         # Black background
#         all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
#     else:
#         # Load video using ffmpeg
#         all_frames = []
#         for f in read_video(input_video_path, skip=input_video_skip, limit=limit):
#             all_frames.append(f)
#         effective_length = min(keypoints.shape[0], len(all_frames))
#         all_frames = all_frames[:effective_length]

#         keypoints = keypoints[input_video_skip:]  # todo remove
#         for idx in range(len(poses)):
#             poses[idx] = poses[idx][input_video_skip:]

#         if fps is None:
#             fps = get_fps(input_video_path)

#     if downsample > 1:
#         keypoints = downsample_tensor(keypoints, downsample)
#         all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
#         for idx in range(len(poses)):
#             poses[idx] = downsample_tensor(poses[idx], downsample)
#             trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
#         fps /= downsample

#     initialized = False
#     image = None
#     lines = []
#     points = None

#     text_left_knee = None
#     text_right_knee = None
#     text_left_elbow = None
#     text_right_elbow = None
#     text_left_upperarm = None
#     text_right_upperarm = None
#     text_neck_flexion = None
#     text_neck_sb = None
#     text_neck_rot = None
#     text_trunk_sb = None
#     text_trunk_flexion = None
#     text_trunk_rot = None
#     text_Rabduction = None
#     text_Labduction = None


#     if limit < 1:
#         limit = len(all_frames)
#     else:
#         limit = min(limit, len(all_frames))

#     parents = skeleton.parents()

#     def update_video(i):
#         nonlocal initialized, image, lines, points, text_left_knee, text_right_knee, text_left_elbow,text_right_elbow, text_left_upperarm, text_right_upperarm, text_neck_flexion, text_neck_sb, text_neck_rot, text_trunk_flexion, text_trunk_sb, text_trunk_rot, text_Rabduction, text_Labduction

#         for n, ax in enumerate(ax_3d):
#             ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
#             ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])

#         # Update 2D poses
#         joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
#         colors_2d = np.full(keypoints.shape[1], 'black')
#         colors_2d[joints_right_2d] = 'black'
#         if not initialized:
#             image = ax_in.imshow(all_frames[i], aspect='equal')
#             text_left_knee = ax_in.text(
#                 -150, 40, f"L_Leg: {angles['left_knee'][i]:.0f}°",
#                 color='black', fontsize=7
#             )
#             text_right_knee = ax_in.text(
#                 -150, 80, f"R_Leg: {angles['right_knee'][i]:.0f}°",
#                 color='black', fontsize=7
#             )
#             text_left_elbow = ax_in.text(
#                 -150, 120, f"L_LowerArm: {angles['left_elbow'][i]:.0f}°",
#                 color='black', fontsize=7
#             )
#             text_right_elbow = ax_in.text(
#                 -150, 160, f"R_RightArm: {angles['right_elbow'][i]:.0f}°",
#                 color='black', fontsize=7
#             )
#             text_left_upperarm = ax_in.text(
#                 -150, 200, f"LUpperArm: {angles['left_upperarm'][i]:.0f}°",
#                 color='black', fontsize=7
#             )
#             text_right_upperarm = ax_in.text(
#                 -150, 240, f"RUpperArm: {angles['right_upperarm'][i]:.0f}°",
#                 color='black', fontsize=7
#             )
#             text_neck_flexion = ax_in.text(
#                 -150, 280, f"NeckFlexion: {angles['neck_flexion'][i]:.0f}°",
#                 color='black', fontsize=7
#             )
#             text_trunk_flexion = ax_in.text(
#                 -150, 320, f"TrunkFlexion: {angles['trunk_flexion'][i]:.0f}°",
#                 color='black', fontsize=7
#             )
#             text_neck_sb = ax_in.text(
#                 -150, 360, f"NeckSB: {angles['neck_sb'][i]:.0f}°",
#                 color='black', fontsize=7
#             )
#             text_neck_rot = ax_in.text(
#                 -150, 400, f"NeckRot: {angles['neck_rot'][i]:.0f}°",
#                 color='black', fontsize=7
#             )
#             text_trunk_sb = ax_in.text(
#                 -150, 440, f"TrunkSB: {angles['trunk_sb'][i]:.0f}°",
#                 color='black', fontsize=7
#             )
#             text_trunk_rot = ax_in.text(
#                 -150, 480, f"TrunkRot: {angles['trunk_rot'][i]:.0f}°",
#                 color='black', fontsize=7
#             )
#             text_Rabduction = ax_in.text(
#                 -150, 520, f"R_Abduction: {angles['Rabduction'][i]:.0f}°",
#                 color='black', fontsize=7
#             )
#             text_Labduction = ax_in.text(
#                 -150, 560, f"L_Abduction: {angles['Labduction'][i]:.0f}°",
#                 color='black', fontsize=7
#             )
#             for j, j_parent in enumerate(parents):
#                 if j_parent == -1:
#                     continue

#                 if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
#                     # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
#                     lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
#                                             [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))

#                 col = 'red' if j in skeleton.joints_right() else 'black'
#                 for n, ax in enumerate(ax_3d):
#                     pos = poses[n][i]
#                     lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
#                                                [pos[j, 1], pos[j_parent, 1]],
#                                                [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))

#             points = ax_in.scatter(*keypoints[i].T, 10, color=colors_2d, edgecolors='white', zorder=10)

#             initialized = True
#         else:
#             image.set_data(all_frames[i])

#             # Update  angle texts
#             text_left_knee.set_text(f"L_Leg: {angles['left_knee'][i]:.0f}°")
#             text_right_knee.set_text(f"R_Leg: {angles['right_knee'][i]:.0f}°")
#             text_left_elbow.set_text(f"L_LowerArm: {angles['left_elbow'][i]:.0f}°")
#             text_right_elbow.set_text(f"R_LowerArm: {angles['right_elbow'][i]:.0f}°")
#             text_left_upperarm.set_text(f"L_UpperArm: {angles['left_upperarm'][i]:.0f}°")
#             text_right_upperarm.set_text(f"R_UpperArm: {angles['right_upperarm'][i]:.0f}°")
#             text_neck_flexion.set_text(f"Neck_Flexion: {angles['neck_flexion'][i]:.0f}°")
#             text_trunk_flexion.set_text(f"Trunk_Flexion: {angles['trunk_flexion'][i]:.0f}°")
#             text_neck_sb.set_text(f"Neck_SB: {angles['neck_sb'][i]:.0f}°")
#             text_neck_rot.set_text(f"NeckRot: {angles['neck_rot'][i]:.0f}°")    
#             text_trunk_sb.set_text(f"Trunk_SB: {angles['trunk_sb'][i]:.0f}°")
#             text_trunk_rot.set_text(f"TrunkRot: {angles['trunk_rot'][i]:.0f}°")
#             text_Rabduction.set_text(f"R_Abduction: {angles['Rabduction'][i]}")
#             text_Labduction.set_text(f"L_Abduction: {angles['Labduction'][i]}")


#             for j, j_parent in enumerate(parents):
#                 if j_parent == -1:
#                     continue

#                 if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
#                     lines[j - 1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
#                                              [keypoints[i, j, 1], keypoints[i, j_parent, 1]])

#                 for n, ax in enumerate(ax_3d):
#                     pos = poses[n][i]
#                     lines_3d[n][j - 1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
#                     lines_3d[n][j - 1][0].set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
#                     lines_3d[n][j - 1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='z')

#             points.set_offsets(keypoints[i])

#         print(f'{i}/{limit}      ', end='\r')

#     fig.tight_layout()

#     anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000 / fps, repeat=False)
#     if output.endswith('.mp4'):
#         Writer = writers['ffmpeg']
#         writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
#         anim.save(output, writer=writer)
#     elif output.endswith('.gif'):
#         anim.save(output, dpi=80, writer='imagemagick')
#     else:
#         raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
#     plt.close()


# -----------------------------------------------------------------------------------------------------------------------------
# Finalized version of video construction. 

def render_animation(keypoints, keypoints_metadata, poses, skeleton, fps, bitrate, azim, output, viewport,
                     limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0, angles=None):
    """
    Render an animation with input video (80%) and angle data (20%) split.
    The angle data is displayed in two columns for better readability.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, writers
    import matplotlib.gridspec as gridspec
    import numpy as np

    plt.ioff()
    fig = plt.figure(figsize=(size * 2, size))
    
    # Create grid with 80:20 split
    gs = gridspec.GridSpec(1, 2, width_ratios=[7, 3])
    
    # Create subplots
    ax_video = fig.add_subplot(gs[0])
    ax_data = fig.add_subplot(gs[1])
    
    # Configure video panel
    ax_video.get_xaxis().set_visible(False)
    ax_video.get_yaxis().set_visible(False)
    ax_video.set_axis_off()
    ax_video.set_title('Input Video')

    # Configure data panel
    ax_data.get_xaxis().set_visible(False)
    ax_data.get_yaxis().set_visible(False)
    ax_data.set_axis_off()
    ax_data.set_title('Joint Angles')

    # Decode video
    if input_video_path is None:
        all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        all_frames = []
        for f in read_video(input_video_path, skip=input_video_skip, limit=limit):
            all_frames.append(f)
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]

        keypoints = keypoints[input_video_skip:]

        if fps is None:
            fps = get_fps(input_video_path)

    if downsample > 1:
        keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
        fps /= downsample

    initialized = False
    image = None
    lines = []
    points = None
    text_elements = {}
    
    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))

    parents = skeleton.parents()

    # Define angle labels divided into two columns
    angle_labels = [
        # Column 1
        [('left_knee', 'L_Leg'),
         ('right_knee', 'R_Leg'),
         ('left_elbow', 'L_LowerArm'),
         ('right_elbow', 'R_LowerArm'),
         ('left_upperarm', 'L_UpperArm'),
         ('right_upperarm', 'R_UpperArm')],
        #  ('Rabduction', 'R_Abd')],
        # Column 2
        [('neck_flexion', 'N_Flex'),
         ('neck_sb', 'N_SB'),
         ('neck_rot', 'N_Rot'),
         ('trunk_flexion', 'T_Flex'),
         ('trunk_sb', 'T_SB'),
         ('trunk_rot', 'T_Rot'),
        #  ('Labduction', 'L_Abd')]
        ]
    ]

    def update_video(i):
        nonlocal initialized, image, lines, points, text_elements

        # Define positions for two columns
        x_positions = [0.05, 0.7]  # Start positions for left and right columns
        base_y = 0.8  # Start from top
        y_spacing = 0.12  # Increased spacing between lines

        if not initialized:
            # Initialize video display
            image = ax_video.imshow(all_frames[i], aspect='equal')
            
            # Initialize angle texts in two columns
            for col_idx, column in enumerate(angle_labels):
                for row_idx, (angle_key, label) in enumerate(column):
                    y_pos = base_y - (row_idx * y_spacing)
                    text_elements[angle_key] = ax_data.text(
                        x_positions[col_idx], y_pos,
                        f"{label}: {angles[angle_key][i]:.0f}°",
                        color='black',
                        fontsize=8,  
                        transform=ax_data.transAxes,
                        fontweight='bold',
                    )

            # Initialize skeleton visualization
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    lines.append(ax_video.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))

            points = ax_video.scatter(*keypoints[i].T, 10, color='black', edgecolors='white', zorder=10)
            initialized = True
        else:
            # Update video frame
            image.set_data(all_frames[i])

            # Update angle texts
            for column in angle_labels:
                for angle_key, label in column:
                    text_elements[angle_key].set_text(f"{label}: {angles[angle_key][i]:.0f}°")

            # Update skeleton visualization
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    lines[j - 1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                           [keypoints[i, j, 1], keypoints[i, j_parent, 1]])

            points.set_offsets(keypoints[i])

        print(f'{i}/{limit}      ', end='\r')

    # Adjust layout
    plt.subplots_adjust(wspace=0.1)

    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000 / fps, repeat=False)
    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()