import torch, cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *


def circle(radius=3.5, h=0.0, axis='z', t0=0, r=1):
    if axis == 'z':
        return lambda t: [radius * np.cos(r * t + t0), radius * np.sin(r * t + t0), h]
    elif axis == 'y':
        return lambda t: [radius * np.cos(r * t + t0), h, radius * np.sin(r * t + t0)]
    else:
        return lambda t: [h, radius * np.cos(r * t + t0), radius * np.sin(r * t + t0)]


def cross(x, y, axis=0):
    T = torch if isinstance(x, torch.Tensor) else np
    return T.cross(x, y, axis)


def normalize(x, axis=-1, order=2):
    if isinstance(x, torch.Tensor):
        l2 = x.norm(p=order, dim=axis, keepdim=True)
        return x / (l2 + 1e-8), l2

    else:
        l2 = np.linalg.norm(x, order, axis)
        l2 = np.expand_dims(l2, axis)
        l2[l2 == 0] = 1
        return x / l2,


def cat(x, axis=1):
    if isinstance(x[0], torch.Tensor):
        return torch.cat(x, dim=axis)
    return np.concatenate(x, axis=axis)


def look_at_rotation(camera_position, at=None, up=None, inverse=False, cv=False):
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors at and up which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.
    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.
    Input:
        camera_position: 3
        at: 1 x 3 or N x 3  (0, 0, 0) in default
        up: 1 x 3 or N x 3  (0, 1, 0) in default
    """

    if at is None:
        at = torch.zeros_like(camera_position)
    else:
        at = torch.tensor(at).type_as(camera_position)
    if up is None:
        up = torch.zeros_like(camera_position)
        up[2] = -1
    else:
        up = torch.tensor(up).type_as(camera_position)

    z_axis = normalize(at - camera_position)[0]
    x_axis = normalize(cross(up, z_axis))[0]
    y_axis = normalize(cross(z_axis, x_axis))[0]

    R = cat([x_axis[:, None], y_axis[:, None], z_axis[:, None]], axis=1)
    return R


def gen_path(pos_gen, at=(0, 0, 0), up=(0, -1, 0), frames=180):
    c2ws = []
    for t in range(frames):
        c2w = torch.eye(4)
        cam_pos = torch.tensor(pos_gen(t * (360.0 / frames) / 180 * np.pi))
        cam_rot = look_at_rotation(cam_pos, at=at, up=up, inverse=False, cv=True)
        c2w[:3, 3], c2w[:3, :3] = cam_pos, cam_rot
        c2ws.append(c2w)
    return torch.stack(c2ws)


class YourOwnDataset(Dataset):
    def _init_(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1):

        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample
        self.define_transforms()

        self.scene_bbox = torch.tensor([[-20.0, -20.0, -20.0], [20.0, 20.0, 20.0]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.define_proj_mat()

        self.white_bg = True
        self.near_far = [0.1, 40.0]

        # self.render_path = torch.tensor(pos_gen(t * (360.0 / frames) / 180 * np.pi))

        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.downsample = downsample

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth

    def read_meta(self):

        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = int(self.meta['w'] / self.downsample), int(self.meta['h'] / self.downsample)
        self.img_wh = [w, h]
        self.focal_x = 0.5 * w / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        self.focal_y = 0.5 * h / np.tan(0.5 * self.meta['camera_angle_y'])  # original focal length
        self.cx, self.cy = self.meta['cx'], self.meta['cy']

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal_x, self.focal_y], center=[self.cx, self.cy])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal_x, 0, self.cx], [0, self.focal_y, self.cy], [0, 0, 1]]).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []

        img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):  # img_list:#

            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}")
            self.image_paths += [image_path]
            img = Image.open(image_path)

            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.view(-1, w * h).permute(1, 0)  # (h*w, 4) RGBA
            if img.shape[-1] == 4:
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        self.poses = torch.stack(self.poses)
        center = torch.mean(self.scene_bbox, dim=0)
        radius = 4.0
        up = torch.mean(self.poses[:, :3, 1], dim=0).tolist()
        pos_gen = circle(radius=radius, h=-0.2 * up[1], axis='y')
        self.render_path = gen_path(pos_gen, up=up, frames=200)
        self.render_path[:, :3, 3] += center

        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)

        #             self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1, *self.img_wh[::-1],
                                                                  3)  # (len(self.meta['frames]),h,w,3)
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:, :3]

    def world2ndc(self, points, lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def _len_(self):
        return len(self.all_rgbs)

    def _getitem_(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx]  # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img}
        return sample
