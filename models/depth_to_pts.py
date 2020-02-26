import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Depth_to_pts(nn.Module):
    def __init__(self, scale_list, max_high=1, npoints=16384):
        super(Depth_to_pts, self).__init__()
        self.scale_list = scale_list
        self.max_high = max_high
        self.npoints = npoints

    def forward(self, input_data):
        # depth_map.shape = [B, H, W]

        pts0 = self.depth_map_to_pts(input_data, level=0)
        return pts0
        pts1 = self.depth_map_to_pts(input_data, level=1)
        pts2 = self.depth_map_to_pts(input_data, level=2)
        pts3 = self.depth_map_to_pts(input_data, level=3)

        pts = torch.cat((pts0, pts1, pts2, pts3), dim=1)

        return pts

    def depth_map_to_pts(self, input_data, level):
        scale_rate = self.scale_list[int(level)]
        depth_map = input_data['depth_map']
        P2 = input_data['P2']

        sampled = self.depth_map_to_sampled(scale_rate, input_data, level)
        zm = self.zoom_matric(1. / scale_rate, 1. / scale_rate, depth_map.shape[0])
        pts = self.sampled_to_pts(sampled, torch.matmul(zm, P2), scale_rate, input_data)
        return pts

    def depth_map_to_sampled(self, scale_rate, input_data, level):
        """
        generate sampled_depth_map

        Note: level start at 0.
        """

        assert level < len(self.scale_list), 'level should in [1, %d)' % len(self.scale_list)
        zoom_rate = 1 - scale_rate

        depth_map, DBH_x, DBH_y = input_data['depth_map'], input_data['DBH_x'], input_data['DBH_y']

        depth_map.unsqueeze_(dim=1)
        bs, c, h, w = depth_map.shape

        theta = torch.Tensor([[scale_rate, 0., zoom_rate * DBH_x / w - zoom_rate / 2],
                              [0., scale_rate, zoom_rate * DBH_y / h - zoom_rate / 2]]).reshape(1, 2, 3)
        theta.requires_grad = False

        # 用于调节采样多少
        size = torch.Size((bs, c, h, w))
        flowfield = F.affine_grid(theta, size).cuda()

        sampled = F.grid_sample(depth_map, flowfield.to(torch.float32), mode='nearest', padding_mode='border')
        sampled = sampled.squeeze(dim=1)

        return sampled

    def zoom_matric(self, k, m, batch_size):
        zoom_matric = torch.eye(3, 3)
        zoom_matric[0, 0] = k
        zoom_matric[1, 1] = m
        zoom_matric = zoom_matric.cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        return zoom_matric

    def sampled_to_pts(self, sampled, zm_P2, scale_rate, input_data):
        # TODO: 后期重构代码时好好整理一下

        DBH_x, DBH_y, R0, R0_inv, C2V, V2C, start_x, start_y = \
            input_data['DBH_x'], input_data['DBH_y'], input_data['R0'], input_data['R0_inv'], \
            input_data['C2V'], input_data['V2C'], input_data['start_x'], input_data['start_y'],

        bs, h, w = int(sampled.shape[0]), int(sampled.shape[1]), int(sampled.shape[2])
        n = int(h * w)

        c_u_2 = zm_P2[:, 0, 2].unsqueeze(1).repeat(1, n)
        c_v_2 = zm_P2[:, 1, 2].unsqueeze(1).repeat(1, n)
        f_u_2 = zm_P2[:, 0, 0].unsqueeze(1).repeat(1, n)
        f_v_2 = zm_P2[:, 1, 1].unsqueeze(1).repeat(1, n)
        b_x_2 = (zm_P2[:, 0, 3] / (-zm_P2[:, 0, 0])).unsqueeze(1).repeat(1, n)
        b_y_2 = (zm_P2[:, 1, 3] / (-zm_P2[:, 1, 1])).unsqueeze(1).repeat(1, n)

        x_range = torch.arange(0, w)
        y_range = torch.arange(0, h)
        y_idxs, x_idxs = torch.meshgrid(y_range, x_range)
        y_idxs, x_idxs = y_idxs.reshape(-1).cuda(), x_idxs.reshape(-1).cuda()
        depth = sampled[:, y_idxs, x_idxs].unsqueeze(-1)
        # uv_depth = torch.zeros([bs, n, 3])
        x_idxs = x_idxs.unsqueeze(0).repeat(bs, 1).unsqueeze(-1).float()
        y_idxs = y_idxs.unsqueeze(0).repeat(bs, 1).unsqueeze(-1).float()
        uv_depth = torch.cat((x_idxs, y_idxs, depth), dim=-1)  # [B, N, 3]

        # project_image_to_rect
        coeff = torch.Tensor([(1 - scale_rate) / scale_rate]).cuda()
        x_offset = (coeff * DBH_x.float()).repeat(bs).unsqueeze(0)
        x_offset = (x_offset + start_x).repeat(1, n)
        y_offset = (coeff * DBH_y.float()).repeat(bs).unsqueeze(0)
        y_offset = (y_offset + start_y).repeat(1, n)

        x = ((uv_depth[:, :, 0] + x_offset - c_u_2) * uv_depth[:, :, 2]) / f_u_2 + b_x_2  # [B, N]
        y = ((uv_depth[:, :, 1] + y_offset - c_v_2) * uv_depth[:, :, 2]) / f_v_2 + b_y_2
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
        pts_rect = torch.cat((x, y, uv_depth[:, :, 2].unsqueeze(-1)), dim=-1)  # [B, N, 3]

        # # 因为输入图片时就只取下面240个像素，所以这里不需要转成velo系砍去高于1米的点云了
        # # project_rect_to_ref
        # pts_3d_ref = torch.transpose(torch.matmul(R0_inv, torch.transpose(pts_rect, 1, 2)), 1, 2)
        #
        # # project_ref_to_velo
        # pts_3d_ref = torch.cat((pts_3d_ref, torch.ones(bs, n, 1).cuda()), dim=2)  # [B, N, 4]
        # pts_velo = torch.matmul(pts_3d_ref, torch.transpose(C2V, 1, 2))  # [B, N, 3]
        #
        # # disregard points with heights larger than 1
        # valid = pts_velo[:, :, 2] < self.max_high
        #
        # pts_velo = torch.cat((pts_velo, torch.ones(pts_velo.shape[0], pts_velo.shape[1], 1).cuda()), dim=2)
        #
        # # project_velo_to_rect
        # pts_rect = torch.matmul(pts_velo, torch.matmul(torch.transpose(V2C, 1, 2), torch.transpose(R0, 1, 2)))

        pts_rect_sampled = torch.zeros(pts_rect.shape[0], self.npoints, 3).cuda()
        for i in range(pts_rect.shape[0]):
            one_pts_rect = pts_rect[i, :, :]
            # one_pts_rect = one_pts_rect[valid[i]]

            '''
            # torch version
            one_pts_depth = one_pts_rect[:, 2]
            pts_near_flag = one_pts_depth < 40.0

            far_idxs = torch.where(pts_near_flag == 0)[0]
            near_idxs = torch.where(pts_near_flag == 1)[0]
            far_sampled_num = round(self.npoints * 0.2)
            near_sampled_num = round(self.npoints * 0.8)

            far_pts = one_pts_rect[far_idxs]
            near_pts = one_pts_rect[near_idxs]

            if far_idxs.shape[0] < far_sampled_num:
                far_pts_sampled = far_pts
                near_sampled_num = self.npoints - far_idxs.shape[0]
            else:
                far_pts_choice = torch.randperm(far_pts.shape[0])[:far_sampled_num]
                far_pts_sampled = far_pts[far_pts_choice]

            near_pts_choice = torch.randperm(near_pts.shape[0])[:near_sampled_num]
            near_pts_sampled = near_pts[near_pts_choice]

            one_pts_rect = torch.cat((near_pts_sampled, far_pts_sampled), dim=0)
            # random shuffle one_pts_rect
            one_pts_rect = one_pts_rect[torch.randperm(one_pts_rect.shape[0])]
            '''

            one_pts_rect_numpy = one_pts_rect.cpu().detach().numpy()
            one_pts_depth = one_pts_rect_numpy[:, 2]
            pts_near_flag = one_pts_depth < 40.0
            far_idxs = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            far_sampled_num = round(self.npoints * 0.2)
            near_sampled_num = round(self.npoints * 0.8)

            if far_idxs.__len__() < far_sampled_num:
                far_idxs_choice = far_idxs
                near_sampled_num = self.npoints - far_idxs.__len__()
            else:
                far_idxs_choice = np.random.choice(far_idxs, far_sampled_num, replace=False)

            near_idxs_choice = np.random.choice(near_idxs, near_sampled_num, replace=False)

            choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                if len(far_idxs_choice) > 0 else near_idxs_choice
            np.random.shuffle(choice)
            ret_pts_rect = one_pts_rect_numpy[choice, :]
            # one_pts_rect[:, -1] -= 0.5  # translate intensity to [-0.5, 0.5]
            pts_rect_sampled[i, :, :3] = torch.from_numpy(ret_pts_rect).cuda()

        return pts_rect_sampled
