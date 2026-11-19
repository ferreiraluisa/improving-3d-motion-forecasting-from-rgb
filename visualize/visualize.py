import torch
import matplotlib.pyplot as plt

def complete_14_joints_17(input_14,
                          used_joints=[2,3,5,6,7,8,9,10,11,12, 13, 14, 15, 16],
                          local_rank='cpu'):
    future_length = input_14.size()[1]
    output_17 = torch.zeros((17, future_length, 3)).to(local_rank)
    output_17[used_joints, :, :] = input_14
    output_17[0, :, :] = torch.tensor([0.0, 0.0, 0.0]).to(local_rank)
    output_17[1, :, :] = torch.tensor([-1.0, 0.0, 0.0]).to(local_rank)
    output_17[4, :, :] = torch.tensor([1.0, 0.0, 0.0]).to(local_rank)

    return output_17


def helper_plot_3d_2im(inputs_3d_1, inputs_3d_2, name, n_joint=17, axis_limit = 6.0 ):
    if (n_joint ==17) or (n_joint == 14):
        skeleton = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7],
                    [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13],
                    [8, 14], [14, 15], [15, 16]]

        # skeleton = [[0, 1], [1, 2], [2, 3], [3,4], [4,5],
        #             [0, 6], [6, 7], [7, 8], [8,9], [9, 10],
        #             [0,11], [11,12], [12, 13], [13, 14],
        #             [12, 16], [16, 17], [17,18],
        #             [12, 19], [12, 20], [12, 21]]
    elif n_joint == 22:
        skeleton = [[0, 1], [1, 2], [2, 3],
                    [4, 5], [5, 6], [6, 7],
                    [4, 8], [0,8],
                    [8,9], [9,10], [10,11],
                    [9, 12],[12, 13],[13, 14], [14, 15], [15, 16],
                    [9,17], [17,18],[18,19], [19,20], [20, 21]]

    elif n_joint==32:
        skeleton = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                    [0, 6], [6, 7], [7, 8], [8, 9], [9, 10],
                    [0, 12], [12, 13], [13, 14], [14, 15],
                    [13, 16], [16, 17], [17, 18], [18, 19], [19, 21], [21, 22],
                    [13, 25], [25, 26], [26, 27], [27, 29], [29, 30]]
    else:
        print('Erorr!')

    inputs_3d_all_1 = inputs_3d_1.detach().cpu()
    inputs_3d_all_2 = inputs_3d_2.detach().cpu()

    if n_joint == 14:

        inputs_3d_all_1 = complete_14_joints_17(inputs_3d_all_1.permute(1,0,2)).permute(1,0,2)
        inputs_3d_all_2 = complete_14_joints_17(inputs_3d_all_2.permute(1,0,2)).permute(1,0,2)



    # Define fixed axis limits - adjust these values based on your data range
    # axis_limit = 6.0  # You can adjust this value based on your typical pose range

    for i in range(inputs_3d_all_1.shape[0]):
        joints3d_1 = inputs_3d_all_1[i, :, :]
        x1 = joints3d_1[:, 0]
        y1 = joints3d_1[:, 1]
        z1 = joints3d_1[:, 2]
        joints3d_2 = inputs_3d_all_2[i, :, :]
        x2 = joints3d_2[:, 0]
        y2 = joints3d_2[:, 1]
        z2 = joints3d_2[:, 2]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x1, y1, z1, color='r', s=50, label=f'Pred')
        ax.scatter(x2, y2, z2, color='b', s=50, label=f'GT')
        #
        for connection in skeleton:
            joint1, joint2 = connection
            ax.plot([joints3d_1[joint1, 0], joints3d_1[joint2, 0]],
                    [joints3d_1[joint1, 1], joints3d_1[joint2, 1]],
                    [joints3d_1[joint1, 2], joints3d_1[joint2, 2]], 'r-')
        for connection in skeleton:
            joint1, joint2 = connection
            ax.plot([joints3d_2[joint1, 0], joints3d_2[joint2, 0]],
                    [joints3d_2[joint1, 1], joints3d_2[joint2, 1]],
                    [joints3d_2[joint1, 2], joints3d_2[joint2, 2]], 'b-')

        # Set fixed axis limits
        ax.set_xlim([-axis_limit, axis_limit])
        ax.set_ylim([-axis_limit, axis_limit])
        ax.set_zlim([-axis_limit, axis_limit])
        ax.legend()

        # Optional: Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Optional: Set viewing angle to be consistent
        ax.view_init(elev=45, azim=90)  # Adjust these angles as needed
        # ax.view_init(elev=0, azim=0)  # z, y plane
        # ax.view_init(elev=0, azim=90)  # z, x plane
        # plt.show()

        plt.savefig(f'pic/{name}_{i}.png')
        plt.close()  # Add this to free memory


def helper_plot_2d_2im(inputs_2d1, inputs_2d2, name):
    skeleton = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7],
                [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13],
                [8, 14], [14, 15], [15, 16]]

    inputs_2d_all1 = inputs_2d1.detach().cpu()
    inputs_2d_all2 = inputs_2d2.detach().cpu()

    # Define fixed axis limits - adjust these values based on your data range
    axis_limit = 0.5  # You can adjust this value based on your typical pose range

    for i in range(inputs_2d1.shape[0]):
        joints2d1 = inputs_2d_all1[i, :, :]
        x1 = joints2d1[:, 0]
        y1 = joints2d1[:, 1]
        joints2d2 = inputs_2d_all2[i, :, :]
        x2 = joints2d2[:, 0]
        y2 = joints2d2[:, 1]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        ax.scatter(x1, y1, color='r', s=50, label=f'Result')
        ax.scatter(x2, y2, color='r', s=50, label=f'Result')

        for connection in skeleton:
            joint1, joint2 = connection
            ax.plot([joints2d1[joint1, 0], joints2d1[joint2, 0]],
                    [joints2d1[joint1, 1], joints2d1[joint2, 1]],
                    'r-')

        for connection in skeleton:
            joint1, joint2 = connection
            ax.plot([joints2d2[joint1, 0], joints2d2[joint2, 0]],
                    [joints2d2[joint1, 1], joints2d2[joint2, 1]],
                    'b-')

        # # Set fixed axis limits
        # ax.set_xlim([-axis_limit, axis_limit])
        # ax.set_ylim([-axis_limit, axis_limit])

        # Optional: Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # Optional: Set viewing angle to be consistent

        plt.savefig(f'pic_est/{name}_{i}.png')
        plt.close()  # Add this to free memory